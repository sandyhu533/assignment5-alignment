# GRPO experiment summary

All runs: OLMo-2-0425-1B, GSM8K, 100 rollout steps (group_size 8, rollout_batch 256). Off-policy uses train_batch 8.


## onpolicy_variants

| experiment | final val_reward | best val_reward | final val_format | final train_reward | val resp_len | last step |
|---|---|---|---|---|---|---|
| GRPO (baseline) | 0.479 | 0.479 | 0.9385 | 0.4609 | 143.1 | 100 |
| Dr_GRPO | 0.4824 | 0.4824 | 0.9639 | 0.4102 | 129.4 | 100 |
| GRPO_constant | 0.4365 | 0.4521 | 0.9404 | 0.4062 | 108.0 | 100 |
| MaxRL | 0.4668 | 0.4727 | 0.9775 | 0.3555 | 124.7 | 100 |
| RFT | 0.4316 | 0.4316 | 0.9775 | 0.2461 | 120.8 | 100 |

## onpolicy_lr_sweep

| experiment | final val_reward | best val_reward | final val_format | final train_reward | val resp_len | last step |
|---|---|---|---|---|---|---|
| lr 3e-6 | 0.0195 | 0.0195 | 0.9775 | 0.0039 | 63.7 | 100 |
| lr 1e-5 (default) | 0.479 | 0.479 | 0.9385 | 0.4609 | 143.1 | 100 |
| lr 3e-5 | 0.0029 | 0.3408 | 0.1094 | 0.0 | 458.5 | 80 |

## onpolicy_prompt

| experiment | final val_reward | best val_reward | final val_format | final train_reward | val resp_len | last step |
|---|---|---|---|---|---|---|
| question_only | 0.0 | 0.0 | 0.0 | 0.0 | 512.0 | 100 |
| r1_zero (default) | 0.479 | 0.479 | 0.9385 | 0.4609 | 143.1 | 100 |
| r1_zero_three_shot | 0.5137 | 0.5166 | 0.9736 | 0.4102 | 120.0 | 100 |

## offpolicy

| experiment | final val_reward | best val_reward | final val_format | final train_reward | val resp_len | last step |
|---|---|---|---|---|---|---|
| naive | 0.4873 | 0.5 | 0.9805 | 0.4492 | 182.6 | 100 |
| noclip | 0.501 | 0.501 | 0.9189 | 0.4219 | 146.5 | 100 |
| clip | 0.4648 | 0.5098 | 0.834 | 0.375 | 108.1 | 100 |
| gspo | 0.5059 | 0.5059 | 0.9863 | 0.4023 | 123.5 | 100 |
