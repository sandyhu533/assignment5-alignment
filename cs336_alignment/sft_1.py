from typing import Callable, Literal
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from torch.nn import functional as F
from torch.optim import Optimizer

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    concated_ids = []
    masks = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt)
        output_ids = tokenizer.encode(output)
        ids = prompt_ids + output_ids
        concated_ids.append(torch.tensor(ids, dtype=torch.long))
        mask = [0] * (len(prompt_ids))
        mask.extend([1] * len(output_ids))
        masks.append(torch.tensor(mask, dtype=torch.bool))
        
    concated = torch.nn.utils.rnn.pad_sequence(concated_ids, True, 0, "right")
    input_tensor = concated[:,:-1]
    label_tensor = concated[:,1:]
    mask_tensor = torch.nn.utils.rnn.pad_sequence(masks, True, 0, "right")[:,1:]
    
    return {
        "input_ids": input_tensor,
        "labels": label_tensor,
        "response_mask": mask_tensor,
    }

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
    logits:torch.Tensor = model(input_ids=input_ids).logits
    log_probs_all = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    result = {"log_probs": gathered}
    if return_token_entropy:
        p = log_probs_all.exp()
        result["token_entropy"] = -(p * log_probs_all).sum(-1)
    return result

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    ) -> tuple[torch.Tensor, dict[str, float]]:
    
    rewards = []
    format_rewards = []
    answer_rewards = []
    for rollout_res, ground_truth in zip(rollout_responses, repeated_ground_truths):
        dic = reward_fn(rollout_res, ground_truth)
        reward = dic["reward"]
        format_reward = dic["format_reward"]
        answer_reward = dic["answer_reward"]
        rewards.append(reward)
        format_rewards.append(format_reward)
        answer_rewards.append(answer_reward)
    
    rew_tensor = torch.tensor(rewards)
    metadata = {
        "mean_reward": sum(rewards)/len(rewards),
        "mean_format_reward": sum(format_rewards)/len(format_rewards),
        "mean_answer_reward": sum(answer_rewards)/len(answer_rewards)
    }
    return rew_tensor, metadata

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    rollout_batch_size = raw_rewards.size(-1)
    grouped = raw_rewards.reshape((rollout_batch_size//group_size, group_size))
    if baseline == "mean":
        baseline_val = grouped.mean(dim=-1, keepdim=True)
    else:
        baseline_val = 0
    if advantage_normalizer == "std":
        normalizer = grouped.std(dim=-1, keepdim=True)+advantage_eps
    elif advantage_normalizer == "mean":
        normalizer = grouped.mean(dim=-1, keepdim=True)+advantage_eps
    else:
        normalizer = 1
    advantages = ((grouped-baseline_val)/normalizer).reshape((rollout_batch_size,))
    return advantages, raw_rewards, {
        "mean": advantages.mean(dim=-1).item(),
        "std": advantages.std(dim=-1).item(),
        "max": advantages.max(dim=-1),
        "min": advantages.min(dim=-1),
    }

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raw_rewards_or_advantages = raw_rewards_or_advantages.reshape(-1,1)
    metadata = {}
    if importance_reweighting_method == "none":
        return -raw_rewards_or_advantages * policy_log_probs, metadata
    elif importance_reweighting_method == "noclip":
        reweight = (policy_log_probs-old_log_probs).exp()
        per_token_loss = raw_rewards_or_advantages * reweight
        return -per_token_loss, metadata
    elif importance_reweighting_method == "grpo":
        reweight = (policy_log_probs-old_log_probs).exp()
        cliped = reweight.clip(1-cliprange, 1+cliprange)
        per_token_loss = torch.minimum(
            raw_rewards_or_advantages*reweight, raw_rewards_or_advantages*cliped
        )
        return -per_token_loss, metadata
    else:
        log_ratio = policy_log_probs - old_log_probs
        mean_log_ratio = (log_ratio * response_mask).sum(dim=-1) / response_mask.sum(dim=-1)
        r_seq = mean_log_ratio.exp().unsqueeze(-1)
        clip_r_seq = r_seq.clip(1-cliprange, 1+cliprange)
        per_token_loss = torch.minimum(
            raw_rewards_or_advantages*r_seq,
            raw_rewards_or_advantages*clip_r_seq
        ).expand_as(policy_log_probs)
        return -per_token_loss, metadata
        

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
    ) -> torch.Tensor:
    per_seq_sum = (per_token_policy_gradient_loss * mask).sum(dim=-1)
    if loss_normalization == "sequence":
        per_seq_cnt = mask.sum(dim=-1)
        per_seq_loss = per_seq_sum / per_seq_cnt
        return per_seq_loss.mean()
    else:
        total_seq_sum = per_seq_sum.sum(-1)
        return total_seq_sum / normalization_constant

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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids, labels, response_mask = tokenized["input_ids"], tokenized["labels"], tokenized["response_mask"]
    raw_rewards, _ = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantage, _, _ = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)
    total_loss = torch.zeros((1,), dtype=torch.float32)
    
    microbatch_size = len(input_ids) // gradient_accumulation_steps
    for i in range(0, len(input_ids), microbatch_size):
        inputs_microbatch = input_ids[i:i+microbatch_size]
        labels_microbatch = labels[i:i+microbatch_size]
        advantage_microbatch = advantage[i:i+microbatch_size]
        response_mask_microbatch = response_mask[i:i+microbatch_size]
        old_log_probs_microbatch = old_log_probs[i:i+microbatch_size] if old_log_probs is not None else None
        response = get_response_log_probs(model, inputs_microbatch, labels_microbatch)
        log_probs = response["log_probs"]
        per_token_loss, _ = compute_policy_gradient_loss(advantage_microbatch, log_probs, importance_reweighting_method, old_log_probs_microbatch, cliprange, response_mask_microbatch)
        loss = aggregate_loss_across_microbatch(per_token_loss, response_mask_microbatch, loss_normalization, normalization_constant)
        if loss_normalization == "sequence":
            loss *= (len(inputs_microbatch) / len(input_ids))
        total_loss += loss.detach()
        loss.backward()
    
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(True), max_grad_norm)
    
    optimizer.step()
    optimizer.zero_grad()
    
    return total_loss, {}