#!/usr/bin/env bash
#
# Run all on-policy GRPO experiments from assignment 5 (sections 4 & 5).
#
#   1. grpo_experiments_standard_on_policy   standard GRPO, r1_zero       (4 seeds)
#   2. grpo_learning_rate                    LR sweep: 3e-6 and 3e-5      (1 seed)
#   3. grpo_prompt_ablation                  question_only, 3-shot        (1 seed)
#   4. grpo_experiments_variants_on_policy   GRPO_constant/DrGRPO/RFT/MaxRL (1 seed)
#
# The default-LR / r1_zero baseline for experiments 2 and 3 is reused from
# experiment 1 (seed 0), so we do not re-run it here.
#
# Runs are sequential (single GPU pair, shared vLLM port). A failed run is
# logged and skipped; the rest continue. A summary is printed at the end.
#
# Usage:  bash scripts/run_on_policy.sh
set -u -o pipefail

cd "$(dirname "$0")/.." || exit 1

# ---- common hyperparameters (shared across every run) ----------------------
RUN="uv run python -m cs336_alignment.grpo_training"
STEPS="${STEPS:-100}"
COMMON=(
  --model allenai/OLMo-2-0425-1B
  --train-path data/gsm8k/train.jsonl
  --test-path data/gsm8k/test.jsonl
  --num-rollout-steps "$STEPS"
  --rollout-batch-size 256
  --train-batch-size 256
  --group-size 8
  --gradient-accumulation-steps 64
  --sampling-temperature 1.0
  --sampling-max-tokens 512
  --max-grad-norm 1.0
  --n-train-examples 6400
  --n-val-examples 1024
  --eval-interval 10
  --vllm-port 8000
)
ROOT=grpo_runs/onpolicy

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

# === 1. Standard on-policy GRPO (4 seeds, default lr=1e-5, r1_zero) ==========
# This is the >=25% deliverable AND the baseline reused by exp 2 & 3.
# run "standard_grpo (4 seeds)" "standard" \
#   --seeds 0 1 2 3 \
#   --learning-rate 1e-5 \
#   --prompt r1_zero \
#   --baseline mean --advantage-normalizer std --loss-normalization sequence

# # === 2. Learning-rate sweep (1 seed each; default point reused from exp 1) ===
# run "lr_3e-6" "lr_sweep/lr_3e-6" \
#   --seeds 0 --learning-rate 3e-6 \
#   --prompt r1_zero \
#   --baseline mean --advantage-normalizer std --loss-normalization sequence

# run "lr_3e-5" "lr_sweep/lr_3e-5" \
#   --seeds 0 --learning-rate 3e-5 \
#   --prompt r1_zero \
#   --baseline mean --advantage-normalizer std --loss-normalization sequence

# # === 3. Prompt ablation (1 seed each; r1_zero baseline reused from exp 1) ====
# run "prompt_question_only" "prompt/question_only" \
#   --seeds 0 --learning-rate 1e-5 \
#   --prompt question_only \
#   --baseline mean --advantage-normalizer std --loss-normalization sequence

# run "prompt_r1_zero_three_shot" "prompt/r1_zero_three_shot" \
#   --seeds 0 --learning-rate 1e-5 \
#   --prompt r1_zero_three_shot \
#   --baseline mean --advantage-normalizer std --loss-normalization sequence

# === 4. RL algorithm variants (1 seed each; r1_zero, default lr) =============
# GRPO_constant: mean / std  / constant
run "variant_GRPO_constant" "variants/GRPO_constant" \
  --seeds 0 --learning-rate 1e-5 --prompt r1_zero \
  --baseline mean --advantage-normalizer std --loss-normalization constant

# Dr_GRPO: mean / none / constant
run "variant_Dr_GRPO" "variants/Dr_GRPO" \
  --seeds 0 --learning-rate 1e-5 --prompt r1_zero \
  --baseline mean --advantage-normalizer none --loss-normalization constant

# RFT: none / none / constant
run "variant_RFT" "variants/RFT" \
  --seeds 0 --learning-rate 1e-5 --prompt r1_zero \
  --baseline none --advantage-normalizer none --loss-normalization constant

# MaxRL: mean / mean / constant
run "variant_MaxRL" "variants/MaxRL" \
  --seeds 0 --learning-rate 1e-5 --prompt r1_zero \
  --baseline mean --advantage-normalizer mean --loss-normalization constant

# ---- summary ---------------------------------------------------------------
echo ""
echo "############################################################"
echo "# SUMMARY"
echo "############################################################"
for line in "${SUMMARY[@]}"; do echo "  $line"; done
echo ""
echo "Results under ${ROOT}/  (each run in a timestamped subdir with config.json + plots/)"
