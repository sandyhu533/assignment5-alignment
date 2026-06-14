from typing import Callable, Literal
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from torch.nn import functional as F
from torch.optim import Optimizer

def tokenize_prompt_and_output(
    prompt_strs: list[str], # list[str] List of prompt strings
    output_strs: list[str], # list[str] List of output strings
    tokenizer: PreTrainedTokenizer,
    ) -> dict[str, torch.Tensor]:
    input_ids = []
    masks = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_id = tokenizer.encode(prompt)
        output_id = tokenizer.encode(output)
        mask = [0 if i < len(prompt_id) else 1 for i in range(len(prompt_id)+len(output_id))]
        masks.append(torch.tensor(mask, dtype=torch.bool))
        prompt_id.extend(output_id)
        input_ids.append(torch.tensor(prompt_id, dtype=torch.long))
    input_tensor = torch.nn.utils.rnn.pad_sequence(input_ids, True, 0, "right")
    mask_tensor = torch.nn.utils.rnn.pad_sequence(masks, True, 0, "right")
    return {
        "input_ids": input_tensor[...,:-1], # shape (batch_size, sequence_length)
        "labels": input_tensor[...,1:], # shape (batch_size, sequence_length)
        "response_mask": mask_tensor[...,1:] # shape (batch_size, sequence_length)
    }

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor, # shape (batch_size, sequence_length)
    labels: torch.Tensor, # shape (batch_size, sequence_length)
    return_token_entropy: bool = False, # If True, also return per-token entropy
    ) -> dict[str, torch.Tensor]:
    logits: torch.Tensor = model(input_ids=input_ids).logits  # (B, T, V)
    log_probs_all = F.log_softmax(logits, dim=-1)
    del logits  # free the large (B, T, V) logits tensor before allocating more
    log_probs = torch.gather(log_probs_all, -1, labels.unsqueeze(-1)).squeeze(-1)
    res = {"log_probs": log_probs}
    if return_token_entropy:
        with torch.no_grad():
            # entr(x) = -x*ln(x); avoids materializing a second (B,T,V) product tensor
            res["token_entropy"] = torch.special.entr(log_probs_all.exp()).sum(-1)
    return res

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    ) -> tuple[torch.Tensor, dict[str, float]]:
    
    raw_rewards = []
    format_rewards = []
    answer_rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_res = reward_fn(response, ground_truth)
        reward = reward_res["reward"]
        format_reward = reward_res["format_reward"]
        answer_reward = reward_res["answer_reward"]
        raw_rewards.append(reward)
        format_rewards.append(format_reward)
        answer_rewards.append(answer_reward)
    metadata = {
            "mean_raw_rewards": sum(raw_rewards)/len(format_rewards),
            "mean_format_reward": sum(format_rewards)/len(format_rewards),
            "mean_answer_reward": sum(answer_rewards)/len(answer_rewards),
        }
    return torch.tensor(raw_rewards), metadata # (rollout_batch_size,)

# done
def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor, # (rollout_batch_size,)
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    ):
    batch_size = raw_rewards.size(-1)
    group_cnt = batch_size // group_size
    group_rewards = raw_rewards.reshape(group_cnt, group_size)
    baseline_val = 0
    if baseline == "mean":
        baseline_val = group_rewards.mean(dim=-1, keepdim=True)
    advantage_val = 1
    if advantage_normalizer == "mean":
        advantage_val = group_rewards.mean(dim=-1, keepdim=True)+advantage_eps
    elif advantage_normalizer == "std":
        advantage_val = group_rewards.std(dim=-1, keepdim=True)+advantage_eps
    advantages = ((group_rewards-baseline_val)/advantage_val).reshape(batch_size,)
    metadata = {
        "mean_grouped_advantages": advantages.mean().detach(),
        "std_grouped_advantages": advantages.std().detach(),
        "max_grouped_advantages": advantages.max().detach(),
        "min_grouped_advantages": advantages.min().detach(),
        "mean_reward": raw_rewards.mean().detach(),
        "std_reward": raw_rewards.std().detach(),
        "max_reward": raw_rewards.max().detach(),
        "min_reward": raw_rewards.min().detach(),
    }
    return advantages, raw_rewards, metadata

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, # (batch_size,)
    policy_log_probs: torch.Tensor, # (batch_size, sequence_length)
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,  # (batch_size, sequence_length)
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None, # (batch_size, sequence_length)
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raw_rewards_or_advantages = raw_rewards_or_advantages.view(-1, 1) # (batch_size, 1)
    metadata = {}
    if importance_reweighting_method == "none":
        return -policy_log_probs*raw_rewards_or_advantages, metadata
    elif importance_reweighting_method == "noclip":
        reweight = (policy_log_probs-old_log_probs).exp()
        return -reweight*raw_rewards_or_advantages, metadata
    elif importance_reweighting_method == "grpo":
        reweight = (policy_log_probs-old_log_probs).exp()
        clipped = reweight.clip(1-cliprange, 1+cliprange)
        per_token_loss = torch.minimum(reweight*raw_rewards_or_advantages,
                                       clipped*raw_rewards_or_advantages)
        # clip fraction: masked tokens whose importance weight got clipped.
        with torch.no_grad():
            was_clipped = (reweight < 1-cliprange) | (reweight > 1+cliprange)
            metadata["clipped_tokens"] = (was_clipped & response_mask.bool()).sum().detach()
        return -per_token_loss, metadata
    else: # gspo
        log_ratio = policy_log_probs-old_log_probs
        mean_log_ratio = (log_ratio*response_mask).sum(dim=-1)/response_mask.sum(-1)
        reweight = mean_log_ratio.exp().unsqueeze(-1)
        clipped = reweight.clip(1-cliprange, 1+cliprange)
        per_token_loss = torch.minimum(reweight*raw_rewards_or_advantages,
                                       clipped*raw_rewards_or_advantages).expand_as(log_ratio)
        # Sequence-level reweight; a clipped sequence counts all its masked tokens.
        with torch.no_grad():
            was_clipped = (reweight < 1-cliprange) | (reweight > 1+cliprange)  # (B,1)
            metadata["clipped_tokens"] = (was_clipped & response_mask.bool()).sum().detach()
        return -per_token_loss, metadata

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor, # (batch_size, sequence_length)
    mask: torch.Tensor, # (batch_size, sequence_length)
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
    ) -> torch.Tensor: # scala
    per_seq_sum = (per_token_policy_gradient_loss * mask).sum(dim=-1) # (batch_size, )
    if loss_normalization == "sequence":
        per_seq_cnt = mask.sum(dim=-1)
        per_seq_mean = per_seq_sum / per_seq_cnt
        return per_seq_mean.mean(dim=-1)
    else:
        total_sum = per_seq_sum.sum(dim=-1)
        return total_sum / normalization_constant

def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
    # Debugging
    debug_memory: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    
    device = next(model.parameters()).device
    microbatch_size = len(repeated_prompts) // gradient_accumulation_steps

    tok_res = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = tok_res["input_ids"].to(device)
    labels = tok_res["labels"].to(device)
    response_mask = tok_res["response_mask"].to(device)

    batch_size, padded_seq_len = input_ids.shape
    # Average non-padding tokens per sequence (response portion only).
    avg_response_tokens = response_mask.float().sum(-1).mean().item()
    
    if old_log_probs is not None:
        old_log_probs = old_log_probs[:, :padded_seq_len]

    # Rewards/advantages computed ONCE over the full batch so group normalization
    # always uses complete groups (independent of gradient_accumulation_steps).
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )
    advantages, raw_rewards, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )
    advantages = advantages.to(device)

    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    entropy_sum = torch.zeros((), dtype=torch.float32, device=device)
    token_count = torch.zeros((), dtype=torch.float32, device=device)
    clip_token_sum = torch.zeros((), dtype=torch.float32, device=device)

    torch.cuda.reset_peak_memory_stats(device)

    for i in range(0, len(input_ids), microbatch_size):
        input_ids_b = input_ids[i:i+microbatch_size]
        labels_b = labels[i:i+microbatch_size]
        mask_b = response_mask[i:i+microbatch_size]
        advantages_b = advantages[i:i+microbatch_size]
        # All-zero advantages contribute zero gradient (e.g. std-normalized group
        # with identical rewards); skip the forward/backward for this microbatch.
        if not advantages_b.any():
            continue
        old_log_probs_b = None
        if old_log_probs is not None:
            old_log_probs_b = old_log_probs[i:i+microbatch_size]
        log_probs_res = get_response_log_probs(model, input_ids_b, labels_b, True)
        log_probs, token_entropy = log_probs_res["log_probs"], log_probs_res["token_entropy"]
        # Entropy is logging-only; detach so we don't pin this microbatch's autograd
        # graph in entropy_sum across the whole accumulation loop (would retain all
        # gradient_accumulation_steps microbatch graphs at once -> OOM).
        entropy_sum += (token_entropy.detach() * mask_b).sum()
        token_count += mask_b.sum()
        per_token_loss, pg_meta = compute_policy_gradient_loss(advantages_b, log_probs, importance_reweighting_method, old_log_probs_b, cliprange, mask_b)
        clip_token_sum += pg_meta.get("clipped_tokens", 0)
        loss = aggregate_loss_across_microbatch(per_token_loss, mask_b, loss_normalization, normalization_constant)
        if loss_normalization == "sequence":
            loss *= (len(input_ids_b) / len(input_ids))
        total_loss += loss.detach()

        # Pre-backward alloc captures the forward peak (logits + activations still
        # resident); post-backward alloc shows the flat baseline once activations free.
        pre_bw_alloc = torch.cuda.memory_allocated(device) / 1024 ** 3

        loss.backward()

        if debug_memory:
            # Healthy: post-bw alloc stays flat across microbatches. Monotonic climb =>
            # something is retaining the per-microbatch autograd graph (a leak).
            mb_idx = i // microbatch_size
            print(
                f"    [mb {mb_idx}] pre_bw_alloc={pre_bw_alloc:.2f}GB "
                f"post_bw_alloc={torch.cuda.memory_allocated(device)/1024**3:.2f}GB "
                f"reserved={torch.cuda.memory_reserved(device)/1024**3:.2f}GB"
            )

    peak_mem_gb = round(torch.cuda.max_memory_allocated(device) / 1024 ** 3, 2)

    clip_value = max_grad_norm if max_grad_norm is not None else float("inf")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

    optimizer.step()
    optimizer.zero_grad()

    metadata = {
        "loss": total_loss,
        "grad_norm": grad_norm,
        "token_entropy": entropy_sum / token_count.clamp(min=1),
        "clip_fraction": (clip_token_sum / token_count.clamp(min=1)),
        "train_reward": reward_metadata["mean_raw_rewards"],
        "train_format_reward": reward_metadata["mean_format_reward"],
        "train_answer_reward": reward_metadata["mean_answer_reward"],
        # Tensor shape / memory diagnostics
        "batch_size": batch_size,
        "padded_seq_len": padded_seq_len,
        "avg_response_tokens": avg_response_tokens,
        "peak_mem_gb": peak_mem_gb,
        **advantage_metadata,
    }
    return total_loss, metadata