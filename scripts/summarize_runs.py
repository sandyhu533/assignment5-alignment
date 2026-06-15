#!/usr/bin/env python3
"""Aggregate every finished GRPO run under grpo_runs/{onpolicy,offpolicy} into
comparison tables and combined overlay plots.

Output -> grpo_runs/summary/
  - summary_table.md / summary_table.csv : final-step metrics for every run
  - onpolicy_variants.png, onpolicy_lr_sweep.png, onpolicy_prompt.png,
    offpolicy.png : per-group multi-panel overlays (one line per variant)
  - combined_val_reward.png : headline val_reward curve for the best of each group
"""
import csv
import glob
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "grpo_runs")
OUT = os.path.join(ROOT, "summary")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Run discovery + loading
# ---------------------------------------------------------------------------
def latest_run(exp_dir):
    """Pick the newest timestamped run dir under an experiment leaf."""
    runs = sorted(d for d in glob.glob(os.path.join(exp_dir, "*")) if os.path.isdir(d))
    return runs[-1] if runs else None


def load_series(run_dir, metric):
    """step -> value, averaged across all seeds present in the run dir."""
    per_seed = []
    for mpath in sorted(glob.glob(os.path.join(run_dir, "seed_*", "metrics.jsonl"))):
        s = {}
        with open(mpath) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("step") == "summary":
                    continue
                v = rec.get(metric)
                if v is not None and v == v:  # skip None / NaN
                    s[rec["step"]] = v
        if s:
            per_seed.append(s)
    if not per_seed:
        return {}
    steps = sorted(set().union(*[set(s) for s in per_seed]))
    return {st: statistics.mean([s[st] for s in per_seed if st in s]) for st in steps}


def final(series):
    return series[max(series)] if series else float("nan")


def peak(series):
    return max(series.values()) if series else float("nan")


# label -> experiment leaf dir, grouped.  "standard" (the canonical on-policy
# GRPO baseline) is reused as the default point in the lr / prompt sweeps.
STD = os.path.join(ROOT, "onpolicy", "standard")
GROUPS = {
    "onpolicy_variants": {
        "GRPO (baseline)": STD,
        "Dr_GRPO": os.path.join(ROOT, "onpolicy/variants/Dr_GRPO"),
        "GRPO_constant": os.path.join(ROOT, "onpolicy/variants/GRPO_constant"),
        "MaxRL": os.path.join(ROOT, "onpolicy/variants/MaxRL"),
        "RFT": os.path.join(ROOT, "onpolicy/variants/RFT"),
    },
    "onpolicy_lr_sweep": {
        "lr 3e-6": os.path.join(ROOT, "onpolicy/lr_sweep/lr_3e-6"),
        "lr 1e-5 (default)": STD,
        "lr 3e-5": os.path.join(ROOT, "onpolicy/lr_sweep/lr_3e-5"),
    },
    "onpolicy_prompt": {
        "question_only": os.path.join(ROOT, "onpolicy/prompt/question_only"),
        "r1_zero (default)": STD,
        "r1_zero_three_shot": os.path.join(ROOT, "onpolicy/prompt/r1_zero_three_shot"),
    },
    "offpolicy": {
        "naive": os.path.join(ROOT, "offpolicy/naive"),
        "noclip": os.path.join(ROOT, "offpolicy/noclip"),
        "clip": os.path.join(ROOT, "offpolicy/clip"),
        "gspo": os.path.join(ROOT, "offpolicy/gspo"),
    },
}

# Resolve every leaf to its newest run dir.
RESOLVED = {g: {lbl: latest_run(d) for lbl, d in m.items()} for g, m in GROUPS.items()}

PANELS = [
    "val_reward", "train_reward",
    "val_format_reward", "train_format_reward",
    "token_entropy", "val_avg_response_length",
    "loss", "grad_norm",
]


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def build_table():
    rows = []
    for group, members in RESOLVED.items():
        for label, run_dir in members.items():
            if not run_dir:
                continue
            vr = load_series(run_dir, "val_reward")
            tr = load_series(run_dir, "train_reward")
            rows.append({
                "group": group,
                "experiment": label,
                "final_val_reward": round(final(vr), 4),
                "best_val_reward": round(peak(vr), 4),
                "final_val_format": round(final(load_series(run_dir, "val_format_reward")), 4),
                "final_train_reward": round(final(tr), 4),
                "best_train_reward": round(peak(tr), 4),
                "final_val_resp_len": round(final(load_series(run_dir, "val_avg_response_length")), 1),
                "last_step": max(vr) if vr else -1,
                "run": os.path.relpath(run_dir, ROOT),
            })
    return rows


def write_table(rows):
    cols = ["group", "experiment", "final_val_reward", "best_val_reward",
            "final_val_format", "final_train_reward", "best_train_reward",
            "final_val_resp_len", "last_step", "run"]
    with open(os.path.join(OUT, "summary_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    lines = ["# GRPO experiment summary\n",
             "All runs: OLMo-2-0425-1B, GSM8K, 100 rollout steps "
             "(group_size 8, rollout_batch 256). Off-policy uses train_batch 8.\n"]
    last_group = None
    header = ("| experiment | final val_reward | best val_reward | final val_format "
              "| final train_reward | val resp_len | last step |")
    sep = "|" + "---|" * 7
    for r in rows:
        if r["group"] != last_group:
            lines.append(f"\n## {r['group']}\n")
            lines.append(header)
            lines.append(sep)
            last_group = r["group"]
        lines.append(
            f"| {r['experiment']} | {r['final_val_reward']} | {r['best_val_reward']} "
            f"| {r['final_val_format']} | {r['final_train_reward']} "
            f"| {r['final_val_resp_len']} | {r['last_step']} |")
    with open(os.path.join(OUT, "summary_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_group(group, members):
    avail = {lbl: d for lbl, d in members.items() if d}
    panels = list(PANELS)
    # add clip_fraction for off-policy where it is logged
    if group == "offpolicy":
        panels = panels[:6] + ["clip_fraction", "loss"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.6))
    axes = axes.ravel()
    cmap = plt.get_cmap("tab10")
    # fixed color per experiment so every panel uses the same mapping
    colors = {lbl: cmap(i % 10) for i, lbl in enumerate(avail)}
    handles = {}
    for ax, metric in zip(axes, panels):
        for label, run_dir in avail.items():
            s = load_series(run_dir, metric)
            if not s:
                continue
            xs = sorted(s)
            (ln,) = ax.plot(xs, [s[x] for x in xs], marker="o", ms=3, lw=1.5,
                            color=colors[label], label=label)
            handles.setdefault(label, ln)
        ax.set_title(metric)
        ax.set_xlabel("rollout step")
        ax.grid(alpha=0.3)
    fig.suptitle(group, fontsize=15, fontweight="bold")
    # one shared legend (color key) for the whole figure, across the top
    fig.legend(handles.values(), handles.keys(), loc="upper center",
               ncol=len(handles), fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT, f"{group}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


def plot_combined_val():
    """One headline panel overlaying val_reward of every experiment."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ci = 0
    cmap = plt.get_cmap("tab20")
    for group, members in RESOLVED.items():
        for label, run_dir in members.items():
            if not run_dir:
                continue
            s = load_series(run_dir, "val_reward")
            if not s:
                continue
            xs = sorted(s)
            ax.plot(xs, [s[x] for x in xs], marker="o", ms=3, lw=1.5,
                    color=cmap(ci % 20), label=f"{group.replace('onpolicy_','').replace('offpolicy','off')}: {label}")
            ci += 1
    ax.set_xlabel("rollout step")
    ax.set_ylabel("val_reward")
    ax.set_title("val_reward across all experiments")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=True, title="experiment")
    fig.tight_layout()
    path = os.path.join(OUT, "combined_val_reward.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("wrote", path)


def main():
    rows = build_table()
    write_table(rows)
    print("wrote summary_table.md / .csv with", len(rows), "runs")
    for group, members in RESOLVED.items():
        plot_group(group, members)
    plot_combined_val()


if __name__ == "__main__":
    main()
