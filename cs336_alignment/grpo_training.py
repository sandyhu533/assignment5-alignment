import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Callable

import torch

from cs336_alignment import checkpoint
from cs336_alignment import vllm_utils
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment import sft

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# r1_zero prompt: forces <think> ... </think> <answer> ... </answer> formatting.
R1_ZERO_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "r1_zero.prompt")


def load_prompt_template(path: str = R1_ZERO_PROMPT_PATH) -> str:
    with open(path) as f:
        return f.read()


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
    """Append a few (prompt, response, reward) rollouts so we can read them by eye."""
    with open(path, "a") as fh:
        for prompt, response, gt in zip(prompts[:n_samples], responses[:n_samples], ground_truths[:n_samples]):
            scores = reward_fn(response, gt)
            fh.write(json.dumps({
                "step": step,
                "prompt": prompt,
                "response": response,
                "ground_truth": gt,
                "reward": scores["reward"],
                "format_reward": scores["format_reward"],
                "answer_reward": scores["answer_reward"],
            }) + "\n")


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


def run_seed(args, seed: int, run_dir: str) -> str:
    """Train one GRPO run with the given seed. Writes metrics.jsonl + rollouts.jsonl
    into run_dir and returns the metrics path."""
    print(f"\n===== seed {seed} -> {run_dir} =====")
    set_seed(seed)
    os.makedirs(run_dir, exist_ok=True)

    device = "cuda:0"
    prompt_template = load_prompt_template()
    reward_fn = r1_zero_reward_fn

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    rollouts_path = os.path.join(run_dir, "rollouts.jsonl")
    logger = JsonlLogger(metrics_path)
    open(rollouts_path, "w").close()  # truncate

    # --- Load datasets (subsample to the configured train/val sizes) ---
    train_prompts, train_ground_truths = load_gsm8k(args.train_path, prompt_template)
    test_prompts, test_ground_truths = load_gsm8k(args.test_path, prompt_template)
    train_prompts = train_prompts[: args.n_train_examples]
    train_ground_truths = train_ground_truths[: args.n_train_examples]
    test_prompts = test_prompts[: args.n_val_examples]
    test_ground_truths = test_ground_truths[: args.n_val_examples]
    print(f"Loaded {len(train_prompts)} train / {len(test_prompts)} val examples.")

    # --- Policy model + optimizer (on the training GPU) ---
    model, tokenizer = checkpoint.get_model_and_tokenizer(args.model, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # --- vLLM inference server (separate GPU) for rollouts + eval ---
    server = vllm_utils.VLLMServer(
        model_id=args.model,
        host="localhost",
        port=args.vllm_port,
        gpu=1,
        seed=seed,
        gpu_memory_utilization=0.85,
    )
    server.start()
    server.init_weight_sync(policy_device=device)  # NCCL group for weight push

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

    # --- Baseline eval before any training (step 0) ---
    val_metrics = evaluate(server, test_prompts, test_ground_truths, reward_fn, eval_sampling_params)
    logger.log({"step": 0, **val_metrics})
    print(f"[seed {seed}][step 0] val: {val_metrics}")

    n_prompts_per_step = args.rollout_batch_size // args.group_size

    # --- GRPO training loop ---
    for step in range(1, args.num_rollout_steps + 1):
        # 1) Sample a batch of prompts for this step.
        prompt_idxs = random.sample(range(len(train_prompts)), n_prompts_per_step)
        batch_prompts = [train_prompts[i] for i in prompt_idxs]
        batch_ground_truths = [train_ground_truths[i] for i in prompt_idxs]

        # 2) ROLLOUT: group_size responses per prompt, aligned 1:1 (prompt-major).
        completions = server.generate_completions(batch_prompts, rollout_sampling_params)
        assert len(completions) == n_prompts_per_step * args.group_size, (
            f"expected {n_prompts_per_step * args.group_size} completions, got {len(completions)}"
        )
        repeated_prompts = [p for p in batch_prompts for _ in range(args.group_size)]
        repeated_ground_truths = [gt for gt in batch_ground_truths for _ in range(args.group_size)]
        rollout_responses = [c.text for c in completions]

        # 3) POLICY UPDATE.
        loss, train_metadata = sft.grpo_train_step(
            model, tokenizer, optimizer,
            args.gradient_accumulation_steps,
            args.max_grad_norm,
            reward_fn,
            repeated_prompts,
            rollout_responses,
            repeated_ground_truths,
            args.group_size,
        )

        # Log train metrics (every step).
        logger.log({"step": step, **train_metadata})
        if step % args.print_interval == 0 or step == 1:
            short = {k: round(v.item() if torch.is_tensor(v) else v, 4)
                     for k, v in train_metadata.items()
                     if k in ("loss", "grad_norm", "token_entropy", "train_reward", "train_format_reward")}
            print(f"[seed {seed}][step {step}] train: {short}")

        # 4) WEIGHT SYNC so the next rollout uses the updated policy.
        server.sync_policy_weights(model)

        # --- Periodically dump rollouts to read by eye ---
        if step % args.rollout_log_interval == 0 or step == args.num_rollout_steps:
            log_rollouts(rollouts_path, step, repeated_prompts, rollout_responses,
                         repeated_ground_truths, reward_fn, args.rollout_log_samples)

        # --- Periodic eval ---
        if step % args.eval_interval == 0 or step == args.num_rollout_steps:
            val_metrics = evaluate(server, test_prompts, test_ground_truths, reward_fn, eval_sampling_params)
            logger.log({"step": step, **val_metrics})
            print(f"[seed {seed}][step {step}] val: {val_metrics}")

    server.stop()
    logger.close()

    # Free GPU memory before the next seed.
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
        aggregate_and_plot(metrics_paths, run_root)
        print(f"\nDone. Per-seed metrics + rollouts under {run_root}/seed_*/, plots in {run_root}/plots/")
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="OLMo-2-0425-1B")
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
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    # Logging / eval cadence
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--print-interval", type=int, default=1)
    parser.add_argument("--rollout-log-interval", type=int, default=10)
    parser.add_argument("--rollout-log-samples", type=int, default=8)
    args = parser.parse_args()
    main(args)
