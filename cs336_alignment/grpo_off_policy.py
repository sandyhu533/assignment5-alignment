import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Callable

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from cs336_alignment import checkpoint
from cs336_alignment import vllm_utils
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment import sft

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# Each prompt maps to its template file and the reward fn that grades its outputs.
# r1_zero / r1_zero_three_shot both emit <think>..</think> <answer>..</answer>, so
# they share r1_zero_reward_fn; question_only emits a bare answer (question_only_reward_fn).
PROMPT_REGISTRY = {
    "r1_zero": ("r1_zero.prompt", r1_zero_reward_fn),
    "r1_zero_three_shot": ("r1_zero_three_shot_gsm8k.prompt", r1_zero_reward_fn),
    "question_only": ("question_only.prompt", question_only_reward_fn),
}

R1_ZERO_PROMPT_PATH = os.path.join(PROMPTS_DIR, "r1_zero.prompt")


def load_prompt_template(path: str = R1_ZERO_PROMPT_PATH) -> str:
    with open(path) as f:
        return f.read()


def resolve_prompt(name: str):
    """Return (prompt_template_str, reward_fn) for a registered prompt name."""
    if name not in PROMPT_REGISTRY:
        raise ValueError(f"unknown prompt {name!r}; choices: {list(PROMPT_REGISTRY)}")
    filename, reward_fn = PROMPT_REGISTRY[name]
    return load_prompt_template(os.path.join(PROMPTS_DIR, filename)), reward_fn


def extract_gsm8k_answer(answer_field: str) -> str:
    """GSM8K gold answers look like '...\\n#### 72'. Keep only the final number."""
    return answer_field.split("####")[-1].strip()


def load_gsm8k(path: str, prompt_template: str) -> tuple[list[str], list[str]]:
    """Load a GSM8K jsonl file into (prompts, ground_truths).

    Each line is {"question": ..., "answer": "...#### <gold>"}.
    `prompts` are formatted with the r1_zero template; `ground_truths` are the
    bare final answers expected by `r1_zero_reward_fn`.
    """
    prompts, ground_truths = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            prompts.append(prompt_template.format(question=example["question"]))
            ground_truths.append(extract_gsm8k_answer(example["answer"]))
    return prompts, ground_truths


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Metric / rollout logging
# ---------------------------------------------------------------------------


class Tee:
    """Duplicate stdout/stderr writes to a log file so the console run log is
    persisted alongside the results."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        self._fh.flush()

    def flush(self):
        self._stream.flush()
        self._fh.flush()


class JsonlLogger:
    """Append-only JSONL writer; one flat dict per line (step + metrics)."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "w")

    def log(self, record: dict) -> None:
        # Coerce tensors -> python floats so the record is JSON-serializable.
        clean = {
            k: (v.item() if torch.is_tensor(v) else v)
            for k, v in record.items()
        }
        self._fh.write(json.dumps(clean) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def log_rollouts(
    path: str,
    step: int,
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    n_samples: int,
) -> None:
    """Append rollouts to jsonl: first n_samples in order, plus top n_samples by reward."""
    scored = []
    for prompt, response, gt in zip(prompts, responses, ground_truths):
        scores = reward_fn(response, gt)
        scored.append({
            "step": step,
            "prompt": prompt,
            "response": response,
            "ground_truth": gt,
            "reward": scores["reward"],
            "format_reward": scores["format_reward"],
            "answer_reward": scores["answer_reward"],
        })

    # Random n_samples + top n_samples by reward (deduped, marked separately).
    rand_idxs = set(random.sample(range(len(scored)), min(n_samples, len(scored))))
    top_idxs = sorted(range(len(scored)), key=lambda i: scored[i]["reward"], reverse=True)[:n_samples]

    entries = []
    for i in rand_idxs:
        entries.append(scored[i] | {"sample_type": "random"})
    for i in top_idxs:
        if i not in rand_idxs:
            entries.append(scored[i] | {"sample_type": "top_reward"})

    with open(path, "a") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    server: vllm_utils.VLLMServer,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    eval_sampling_params: dict,
) -> dict[str, float]:
    """Run greedy generation over the val set and score with reward_fn.

    Returns val total/format reward and the average response length (in tokens).
    """
    completions = server.generate_completions(prompts, eval_sampling_params)

    n_correct = 0.0
    n_format = 0.0
    total_response_len = 0
    for completion, ground_truth in zip(completions, ground_truths):
        scores = reward_fn(completion.text, ground_truth)
        n_correct += scores["reward"]
        n_format += scores["format_reward"]
        total_response_len += len(completion.token_ids)

    n = len(prompts)
    return {
        "val_reward": n_correct / n,            # total reward (== accuracy)
        "val_format_reward": n_format / n,      # format reward
        "val_avg_response_length": total_response_len / n,
    }


# ---------------------------------------------------------------------------
# Single-seed training run
# ---------------------------------------------------------------------------


def _fmt(seconds: float) -> str:
    """Format elapsed seconds as e.g. '1m23s' or '45s'."""
    s = int(seconds)
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def compute_old_log_probs(model, repeated_prompts, rollout_responses, tokenizer,
                          device, micro_batch_size):
    """Per-token log-probs of the rollout responses under the *current* model.

    At call time the model still IS pi_old (no update has happened yet), so this
    is the reference policy for the off-policy importance weights. Computed once
    per rollout batch and frozen for the whole inner loop.

    The full batch is tokenized once so every chunk shares the same padded length
    (T_global); the forward is then chunked to avoid materializing a single
    (rollout_batch_size, T, V) logits tensor (OOM).
    """
    tok = sft.tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = tok["input_ids"].to(device)
    labels = tok["labels"].to(device)
    was_training = model.training
    model.eval()
    chunks = []
    with torch.no_grad():
        for j in range(0, input_ids.size(0), micro_batch_size):
            chunks.append(
                sft.get_response_log_probs(
                    model, input_ids[j:j+micro_batch_size], labels[j:j+micro_batch_size]
                )["log_probs"]
            )
    if was_training:
        model.train()
    return torch.cat(chunks, dim=0)  # (rollout_batch_size, T_global)


# Diagnostics where the max across a rollout batch's inner sub-steps is more
# meaningful than the mean; everything else is mean-reduced.
_MAX_REDUCE_METRICS = {"peak_mem_gb", "padded_seq_len", "batch_size"}


def aggregate_substep_metrics(substeps: list[dict]) -> dict[str, float]:
    """Reduce the per-train-step metric dicts of one rollout batch into one dict.

    Off-policy takes many gradient steps per rollout batch; rather than logging
    each (which collides on the wandb step axis), we mean-reduce scalar training
    metrics and max-reduce memory/shape diagnostics into a single per-step record.
    Tensors are converted to floats (also detaching them for logging).
    """
    if not substeps:
        return {}
    out: dict[str, float] = {}
    for k in substeps[0]:
        vals = [m[k].item() if torch.is_tensor(m[k]) else m[k]
                for m in substeps if k in m]
        if not vals:
            continue
        out[k] = max(vals) if k in _MAX_REDUCE_METRICS else sum(vals) / len(vals)
    return out


def run_seed(args, seed: int, run_dir: str) -> str:
    """Train one GRPO run with the given seed. Writes metrics.jsonl + rollouts.jsonl
    into run_dir and returns the metrics path."""
    import time
    seed_start = time.perf_counter()

    def elapsed() -> str:
        return _fmt(time.perf_counter() - seed_start)

    print(f"\n===== seed {seed} -> {run_dir} =====")
    set_seed(seed)
    os.makedirs(run_dir, exist_ok=True)

    device = "cuda:0"
    prompt_template, reward_fn = resolve_prompt(args.prompt)

    # For constant loss normalization (Dr.GRPO/RFT/MaxRL/GRPO_constant), divide the
    # total loss by Z = B*G*L (rollout_batch_size * max generation length) per the
    # handout. Sequence normalization ignores this (passes None).
    normalization_constant = (
        args.rollout_batch_size * args.sampling_max_tokens
        if args.loss_normalization == "constant"
        else None
    )

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    rollouts_path = os.path.join(run_dir, "rollouts.jsonl")
    logger = JsonlLogger(metrics_path)
    # Per-rollout-step records (metrics.jsonl) align with the on-policy runs and
    # drive the plots. Off-policy additionally takes many gradient updates per
    # step; their unaggregated per-update metrics go to a separate file keyed by a
    # monotonic `update` counter, so the per-step plots stay clean.
    updates_path = os.path.join(run_dir, "updates.jsonl")
    update_logger = JsonlLogger(updates_path)
    open(rollouts_path, "w").close()  # truncate

    # --- Load datasets ---
    t0 = time.perf_counter()
    print(f"[seed {seed}][{elapsed()}] loading datasets...")
    train_prompts, train_ground_truths = load_gsm8k(args.train_path, prompt_template)
    test_prompts, test_ground_truths = load_gsm8k(args.test_path, prompt_template)
    train_prompts = train_prompts[: args.n_train_examples]
    train_ground_truths = train_ground_truths[: args.n_train_examples]
    test_prompts = test_prompts[: args.n_val_examples]
    test_ground_truths = test_ground_truths[: args.n_val_examples]
    print(f"[seed {seed}][{elapsed()}] loaded {len(train_prompts)} train / {len(test_prompts)} val examples ({_fmt(time.perf_counter()-t0)})")

    # --- Policy model + optimizer ---
    t0 = time.perf_counter()
    print(f"[seed {seed}][{elapsed()}] loading policy model ({args.model})...")
    model, tokenizer = checkpoint.get_model_and_tokenizer(args.model, device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    print(f"[seed {seed}][{elapsed()}] policy model loaded ({_fmt(time.perf_counter()-t0)})")

    # --- vLLM inference server (skipped in debug mode) ---
    if args.debug_no_vllm:
        print(f"[seed {seed}][{elapsed()}] [DEBUG] skipping vLLM server and NCCL init")
        server = None
    else:
        t0 = time.perf_counter()
        print(f"[seed {seed}][{elapsed()}] starting vLLM server on GPU 1 (port {args.vllm_port})...")
        server = vllm_utils.VLLMServer(
            model_id=args.model,
            host="localhost",
            port=args.vllm_port,
            gpu=1,
            seed=seed,
            gpu_memory_utilization=0.85,
            vllm_log_path=os.path.join(run_dir, "vllm.log"),
        )
        server.start()
        print(f"[seed {seed}][{elapsed()}] vLLM server ready ({_fmt(time.perf_counter()-t0)})")

        t0 = time.perf_counter()
        print(f"[seed {seed}][{elapsed()}] initializing NCCL weight sync...")
        server.init_weight_sync(policy_device=device)
        print(f"[seed {seed}][{elapsed()}] NCCL ready ({_fmt(time.perf_counter()-t0)})")

    rollout_sampling_params = {
        "temperature": args.sampling_temperature,
        "max_tokens": args.sampling_max_tokens,
        "n": args.group_size,
        "seed": seed,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }
    eval_sampling_params = {
        "temperature": 0.0,
        "max_tokens": args.sampling_max_tokens,
        "n": 1,
        "seed": seed,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }

    # --- Baseline eval (step 0) ---
    if args.debug_no_vllm:
        print(f"[seed {seed}][{elapsed()}] [DEBUG] skipping baseline eval")
    else:
        t0 = time.perf_counter()
        print(f"[seed {seed}][{elapsed()}] running baseline eval (step 0)...")
        val_metrics = evaluate(server, test_prompts, test_ground_truths, reward_fn, eval_sampling_params)
        logger.log({"step": 0, **val_metrics})
        print(f"[seed {seed}][{elapsed()}] step 0 val: {val_metrics} ({_fmt(time.perf_counter()-t0)})")

    n_prompts_per_step = args.rollout_batch_size // args.group_size

    # Each train_batch slice must be whole groups, else group-normalized advantages
    # (computed inside grpo_train_step) would be taken over a partial group.
    assert args.rollout_batch_size % args.train_batch_size == 0, \
        "rollout_batch_size must be divisible by train_batch_size"
    assert args.train_batch_size % args.group_size == 0, \
        "train_batch_size must be divisible by group_size (slices must be whole groups)"

    # Per-phase timing accumulators (seconds).
    t_rollout = t_old_probs = t_update = t_sync = t_eval = 0.0
    global_update = 0  # monotonic gradient-update index for fine-grained logging

    # Record allocation history (with stacks) so an OOM can be dumped + visualized.
    if args.debug_memory:
        torch.cuda.memory._record_memory_history(max_entries=100_000)

    # --- GRPO training loop ---
    print(f"[seed {seed}][{elapsed()}] starting GRPO training loop ({args.num_rollout_steps} steps)...")
    for step in range(1, args.num_rollout_steps + 1):
        step_start = time.perf_counter()

        # 1) Sample a batch of prompts.
        prompt_idxs = random.sample(range(len(train_prompts)), n_prompts_per_step)
        batch_prompts = [train_prompts[i] for i in prompt_idxs]
        batch_ground_truths = [train_ground_truths[i] for i in prompt_idxs]

        # 2) ROLLOUT.
        t0 = time.perf_counter()
        repeated_prompts = [p for p in batch_prompts for _ in range(args.group_size)]
        repeated_ground_truths = [gt for gt in batch_ground_truths for _ in range(args.group_size)]
        if args.debug_no_vllm:
            # Dummy responses: valid format so reward_fn and tokenizer both work.
            rollout_responses = [
                "<think>dummy</think><answer>1</answer>"
                for _ in range(n_prompts_per_step * args.group_size)
            ]
        else:
            completions = server.generate_completions(batch_prompts, rollout_sampling_params)
            assert len(completions) == n_prompts_per_step * args.group_size, (
                f"expected {n_prompts_per_step * args.group_size} completions, got {len(completions)}"
            )
            rollout_responses = [c.text for c in completions]
        t_rollout += time.perf_counter() - t0

        # 3) OLD PROBS — reference log-probs for the off-policy importance weights,
        # computed once (chunked to avoid OOM) before any update.
        t0 = time.perf_counter()
        old_log_probs = compute_old_log_probs(
            model, repeated_prompts, rollout_responses, tokenizer,
            device, args.train_batch_size,
        )
        t_old_probs += time.perf_counter() - t0
        
        # 4) POLICY UPDATE — train_batch_size-sized slices (== whole groups), one
        # gradient step each. Many steps per rollout batch (off-policy); their
        # metrics are aggregated and logged once per rollout step.
        model.train()
        substep_metrics = []
        t0 = time.perf_counter()
        for i in range(0, args.rollout_batch_size, args.train_batch_size):
            sl = slice(i, i + args.train_batch_size)
            try:
                _, train_metadata = sft.grpo_train_step(
                    model, tokenizer, optimizer,
                    args.gradient_accumulation_steps,
                    args.max_grad_norm,
                    reward_fn,
                    repeated_prompts[sl],
                    rollout_responses[sl],
                    repeated_ground_truths[sl],
                    args.group_size,
                    baseline=args.baseline,
                    advantage_normalizer=args.advantage_normalizer,
                    loss_normalization=args.loss_normalization,
                    normalization_constant=normalization_constant,
                    debug_memory=args.debug_memory,
                    importance_reweighting_method=args.importance_reweighting_method,
                    cliprange=args.cliprange,
                    old_log_probs=old_log_probs[sl]
                )
            except torch.cuda.OutOfMemoryError:
                # Dump the allocation history (with Python stacks) at the moment of OOM
                # so it can be loaded into https://pytorch.org/memory_viz.
                if args.debug_memory:
                    snap_path = os.path.join(run_dir, "oom_snapshot.pickle")
                    torch.cuda.memory._dump_snapshot(snap_path)
                    print(f"[seed {seed}][step {step}] CUDA OOM -> dumped snapshot to {snap_path}")
                raise
            substep_metrics.append(train_metadata)
            # Fine-grained per-gradient-update record (separate file/axis).
            global_update += 1
            update_logger.log({
                "update": global_update,
                "step": step,
                "substep": i // args.train_batch_size,
                **train_metadata,
            })
        t_update += time.perf_counter() - t0

        # Aggregate the rollout batch's gradient steps into one per-step record.
        step_metrics = aggregate_substep_metrics(substep_metrics)
        step_time = time.perf_counter() - step_start
        step_metrics["step_time"] = round(step_time, 2)
        logger.log({"step": step, **step_metrics})
        if step % args.print_interval == 0 or step == 1:
            short = {k: round(step_metrics[k], 4)
                     for k in ("loss", "grad_norm", "token_entropy", "clip_fraction",
                               "train_reward", "train_format_reward")
                     if k in step_metrics}
            shape_info = (
                f"batch={int(step_metrics['batch_size'])}"
                f" seq={int(step_metrics['padded_seq_len'])}"
                f" resp_tok={step_metrics['avg_response_tokens']:.1f}"
                f" peak_mem={step_metrics['peak_mem_gb']:.2f}GB"
            )
            print(f"[seed {seed}][{elapsed()}][step {step}/{args.num_rollout_steps}] train: {short} | {shape_info} | step={_fmt(step_time)}")

        # 5) WEIGHT SYNC.
        if not args.debug_no_vllm:
            t0 = time.perf_counter()
            server.sync_policy_weights(model)
            t_sync += time.perf_counter() - t0

        # --- Periodic rollout dump ---
        if step == 1 or step % args.rollout_log_interval == 0 or step == args.num_rollout_steps:
            log_rollouts(rollouts_path, step, repeated_prompts, rollout_responses,
                         repeated_ground_truths, reward_fn, args.rollout_log_samples)

        # --- Periodic eval ---
        if not args.debug_no_vllm and (step % args.eval_interval == 0 or step == args.num_rollout_steps):
            t0 = time.perf_counter()
            print(f"[seed {seed}][{elapsed()}][step {step}] running eval...")
            val_metrics = evaluate(server, test_prompts, test_ground_truths, reward_fn, eval_sampling_params)
            dt_eval = time.perf_counter() - t0
            t_eval += dt_eval
            logger.log({"step": step, **val_metrics})
            print(f"[seed {seed}][{elapsed()}][step {step}] val: {val_metrics} ({_fmt(dt_eval)})")

    total_time = time.perf_counter() - seed_start
    print(
        f"\n[seed {seed}] done in {_fmt(total_time)} | "
        f"rollout={_fmt(t_rollout)} old_probs={_fmt(t_old_probs)} update={_fmt(t_update)} "
        f"weight_sync={_fmt(t_sync)} eval={_fmt(t_eval)}"
    )
    logger.log({
        "step": "summary",
        "total_time_s": round(total_time, 2),
        "t_rollout_s": round(t_rollout, 2),
        "t_update_s": round(t_update, 2),
        "t_weight_sync_s": round(t_sync, 2),
        "t_eval_s": round(t_eval, 2),
    })

    if args.debug_memory:
        torch.cuda.memory._record_memory_history(enabled=None)

    if server is not None:
        server.stop()
    logger.close()
    update_logger.close()

    del model, optimizer
    torch.cuda.empty_cache()
    return metrics_path


# ---------------------------------------------------------------------------
# Cross-seed aggregation + plotting
# ---------------------------------------------------------------------------

# Metrics we plot. Train metrics are logged every step; val metrics at eval steps.
PLOT_METRICS = [
    "loss",
    "grad_norm",
    "token_entropy",
    "clip_fraction",
    "train_reward",
    "train_format_reward",
    "val_reward",
    "val_format_reward",
    "val_avg_response_length",
]


def _load_series(metrics_path: str, metric: str) -> dict[int, float]:
    """step -> value for one metric from one run's metrics.jsonl."""
    series = {}
    with open(metrics_path) as f:
        for line in f:
            rec = json.loads(line)
            if metric in rec:
                series[rec["step"]] = rec[metric]
    return series


def aggregate_and_plot(metrics_paths: list[str], out_dir: str) -> None:
    """Read every seed's metrics.jsonl and produce one plot per metric showing the
    cross-seed mean with a ±std band (and min/max envelope)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots. Metrics are in:", metrics_paths)
        return

    import statistics

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for metric in PLOT_METRICS:
        # Gather per-seed series, then align on the shared step grid.
        per_seed = [_load_series(p, metric) for p in metrics_paths]
        per_seed = [s for s in per_seed if s]
        if not per_seed:
            continue
        steps = sorted(set.intersection(*[set(s.keys()) for s in per_seed]))
        if not steps:
            continue

        means, stds, mins, maxs = [], [], [], []
        for step in steps:
            vals = [s[step] for s in per_seed]
            means.append(statistics.mean(vals))
            stds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
            mins.append(min(vals))
            maxs.append(max(vals))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, means, label="mean", color="C0")
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        ax.fill_between(steps, lower, upper, alpha=0.25, color="C0", label="±std")
        ax.fill_between(steps, mins, maxs, alpha=0.10, color="C0", label="min/max")
        ax.set_xlabel("rollout step")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} (n={len(per_seed)} seeds)")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(plots_dir, f"{metric}.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args):
    # Timestamped run dir: grpo_runs/<YYYY-MM-DD_HH-MM-SS>/ holds config.json,
    # one seed_<n>/ subfolder per seed, and the cross-seed plots/.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_root = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_root, exist_ok=True)

    # Record the exact run parameters for reproducibility.
    config = {"timestamp": timestamp, **vars(args)}
    with open(os.path.join(run_root, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Persist the console run log to run.log inside the run dir.
    log_fh = open(os.path.join(run_root, "run.log"), "w")
    sys.stdout = Tee(sys.__stdout__, log_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh)
    print(f"Run dir: {run_root}\nConfig: {config}")

    try:
        metrics_paths = []
        for seed in args.seeds:
            seed_dir = os.path.join(run_root, f"seed_{seed}")
            metrics_paths.append(run_seed(args, seed, seed_dir))
            # Refresh plots after every completed seed so partial results are
            # visible mid-run (n grows from 1 up to len(seeds)).
            aggregate_and_plot(metrics_paths, run_root)
        print(f"\nDone. Per-seed metrics + rollouts under {run_root}/seed_*/, plots in {run_root}/plots/")
    except Exception:
        import traceback
        traceback.print_exc()  # printed while sys.stderr is still Tee'd into run.log
        raise
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--train-path", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--test-path", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="grpo_runs")
    parser.add_argument("--vllm-port", type=int, default=8000)
    # Seeds: 4 runs by default to measure cross-seed variance.
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    # GRPO hyperparameters
    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    # Prompt + RL algorithm variant knobs (default = standard on-policy GRPO)
    parser.add_argument("--prompt", type=str, default="r1_zero",
                        choices=["r1_zero", "r1_zero_three_shot", "question_only"])
    parser.add_argument("--baseline", type=str, default="mean",
                        choices=["mean", "none"])
    parser.add_argument("--importance-reweighting-method", type=str, default="none",
                        choices=["none", "noclip", "grpo", "gspo"])
    parser.add_argument("--cliprange", type=float, default=0.0)
    parser.add_argument("--advantage-normalizer", type=str, default="std",
                        choices=["std", "none", "mean"])
    parser.add_argument("--loss-normalization", type=str, default="sequence",
                        choices=["sequence", "constant"])
    # Debug
    parser.add_argument("--debug-no-vllm", action="store_true",
                        help="Skip vLLM server + NCCL + eval; use dummy rollouts to debug backward/optimizer.")
    parser.add_argument("--debug-memory", action="store_true",
                        help="Print per-microbatch GPU alloc and dump a memory snapshot (memory_viz pickle) on OOM.")
    # Logging / eval cadence
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--print-interval", type=int, default=1)
    parser.add_argument("--rollout-log-interval", type=int, default=10)
    parser.add_argument("--rollout-log-samples", type=int, default=8)
    args = parser.parse_args()
    main(args)
