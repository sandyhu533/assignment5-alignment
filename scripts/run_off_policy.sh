#!/usr/bin/env bash
#
# Run the off-policy GRPO experiments from assignment 5 (section 6.4,
# grpo_experiments_off_policy). Each run is 32x off-policy: one rollout batch of
# 256 is split into train_batch_size=8 slices (== whole groups) and consumed over
# 32 sequential gradient updates, so the policy drifts from the sampling policy
# and importance reweighting / clipping kicks in.
#
#   1. offpolicy_naive   importance_reweighting_method=none    (4 seeds)
#   2. offpolicy_noclip  importance_reweighting_method=noclip  (4 seeds)
#   3. offpolicy_clip    importance_reweighting_method=grpo,  cliprange=0.2   (4 seeds)
#   4. offpolicy_gspo    importance_reweighting_method=gspo,  cliprange=3e-4  (4 seeds)
#
# Hyperparameters are kept fixed w.r.t. the standard on-policy GRPO runs
# (r1_zero prompt, baseline=mean, advantage-normalizer=std,
# loss-normalization=sequence) so the off-policy/on-policy comparison is clean.
#
# Runs are sequential (single GPU pair, shared vLLM port). A failed run is
# logged and skipped; the rest continue. A summary is printed at the end.
#
# Usage:  bash scripts/run_off_policy.sh
#   Overridable via env vars, e.g.:  STEPS=10 LR=3e-6 bash scripts/run_off_policy.sh
set -u -o pipefail

cd "$(dirname "$0")/.." || exit 1

# ---- common hyperparameters (shared across every run) ----------------------
RUN="uv run python -m cs336_alignment.grpo_off_policy"
STEPS="${STEPS:-100}"
# Learning rate: keep equal to your standard GRPO runs (default 1e-5).
LR="${LR:-1e-5}"
# PDF asks for gradient_accumulation_steps=1; we default to 2 (mathematically
# equivalent — same optimizer update, half the per-microbatch activation memory)
# to avoid OOM. Override with GRAD_ACCUM=1 if memory allows.
GRAD_ACCUM="${GRAD_ACCUM:-2}"
COMMON=(
  --model allenai/OLMo-2-0425-1B
  --train-path data/gsm8k/train.jsonl
  --test-path data/gsm8k/test.jsonl
  --num-rollout-steps "$STEPS"
  --learning-rate "$LR"
  # 32x off-policy: rollout 256, train 8 (== one group), 256/8 = 32 updates/batch.
  --rollout-batch-size 256
  --train-batch-size 8
  --gradient-accumulation-steps "$GRAD_ACCUM"
  --group-size 8
  --sampling-temperature 1.0
  --sampling-max-tokens 512
  --max-grad-norm 1.0
  --n-train-examples 6400
  --n-val-examples 1024
  --eval-interval 10
  --vllm-port 8000
  # Standard GRPO advantage estimator (kept fixed across variants).
  --prompt r1_zero
  --baseline mean
  --advantage-normalizer std
  --loss-normalization sequence
)
ROOT=grpo_runs/offpolicy

declare -a SUMMARY

# run <name> <output-subdir> <extra args...>
run () {
  local name=$1; shift
  local subdir=$1; shift
  echo ""
  echo "############################################################"
  echo "# EXPERIMENT: ${name}"
  echo "# args: $*"
  echo "############################################################"
  $RUN "${COMMON[@]}" --output-dir "${ROOT}/${subdir}" "$@"
  local rc=$?
  if [ $rc -eq 0 ]; then SUMMARY+=("OK    ${name}"); else SUMMARY+=("FAIL(${rc}) ${name}"); fi
}

# === 1. offpolicy_naive — no importance reweighting (biased; the baseline) ===
run "offpolicy_naive (4 seeds)" "naive" \
  --seeds 0 1 2 3 \
  --importance-reweighting-method none

# === 2. offpolicy_noclip — token-level importance weights, no clipping =======
run "offpolicy_noclip (4 seeds)" "noclip" \
  --seeds 0 1 2 3 \
  --importance-reweighting-method noclip

# === 3. offpolicy_clip — GRPO/PPO token-level clipping (cliprange 0.2) =======
run "offpolicy_clip (4 seeds)" "clip" \
  --seeds 0 1 2 3 \
  --importance-reweighting-method grpo --cliprange 0.2

# === 4. offpolicy_gspo — GSPO sequence-level clipping (cliprange 3e-4) =======
run "offpolicy_gspo (4 seeds)" "gspo" \
  --seeds 0 1 2 3 \
  --importance-reweighting-method gspo --cliprange 3e-4

# ---- summary ---------------------------------------------------------------
echo ""
echo "############################################################"
echo "# SUMMARY"
echo "############################################################"
for line in "${SUMMARY[@]}"; do echo "  $line"; done
echo ""
echo "Results under ${ROOT}/  (each run in a timestamped subdir with config.json + plots/)"
echo "Per-step metrics: metrics.jsonl (aligned with on-policy);"
echo "per-update metrics: updates.jsonl (fine-grained, 32 updates/step)."
