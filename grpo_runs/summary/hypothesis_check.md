# Results vs. assignment expectations

Each section states the hypothesis/expectation from the handout
(`cs336_spring2026_assignment5_alignment.pdf`) and whether our runs match.

> **Global caveat on confidence.** The handout asks for **4 seeds** on the
> standard, variants, and off-policy experiments and repeatedly asks "how
> confident are you given the between-run variance?". We ran **2 seeds** for
> standard and **1 seed** for every variant / prompt / lr / off-policy run.
> So directional findings below are plausible but **low-confidence**; single-seed
> rankings that are within a few points should be treated as ties.

## 1. Standard on-policy GRPO — ✅ meets bar
- **Expectation (p25):** validation reward improves over training; procedure
  reaches **≥ 25% final val accuracy averaged over seeds**.
- **Result:** val_reward rises monotonically to **0.479** (avg of 2 seeds),
  format reward ~0.94. Comfortably clears the 25% bar. Seeds 0/1 agree (0.478 /
  0.481), so low variance on the one experiment where we have >1 seed.

## 2. Learning-rate sweep — ✅ matches textbook expectation
- **Expectation (p25-26):** sweep one lr below and one above default; lower lr
  trains slower, higher lr may **diverge** ("note divergence if the optimizer
  diverges").
- **Result:** exactly the classic U-shape.
  - `3e-6`: underfits — only **0.020** in 100 steps (too slow).
  - `1e-5`: best — **0.479**.
  - `3e-5`: **diverges** — spikes to 0.34 early, then collapses to ~0;
    format_reward crashes 0.94→0.11, response length blows up to the 512 cap,
    and the run dies at step 80.
- **Verdict:** matches expectation precisely; 1e-5 is the right default.

## 3. Prompt ablation — ✅ directionally as expected (⚠ 1 seed)
- **Expectation (p26):** prompt shapes which rollouts are explored; compare
  question_only and r1_zero_three_shot to zero-shot r1_zero.
- **Result:**
  - `r1_zero_three_shot`: **best, 0.514** (also shorter responses ~120 tok,
    format 0.974) — few-shot format demonstrations help.
  - `r1_zero`: 0.479.
  - `question_only`: **total failure, 0.000** — with no format scaffolding the
    model never emits the required answer format, so format_reward stays ~0,
    reward (which is gated on format) stays 0, and length saturates at 512.
- **Verdict:** consistent with the hypothesis — the reward is tied to the
  r1_zero format, so a prompt that demonstrates it (3-shot) wins and one that
  drops it (question_only) collapses. Can't judge variance with 1 seed.

## 4. On-policy variants — ✅ mostly as expected (⚠ 1 seed)
- **Expectation (p27-31):** Dr. GRPO argues removing std-normalization and
  length-normalization removes bias and should be **competitive with / better
  than** standard GRPO. Handout's tuning caveat: variants reuse the baseline's
  tuned lr, so *beating* the baseline is a real win, but *underperforming* is
  inconclusive (could improve with its own lr).
- **Result (final val_reward):** Dr_GRPO **0.482** ≈ GRPO **0.479** > MaxRL
  0.467 > GRPO_constant 0.437 > RFT 0.432.
  - Dr_GRPO matches/edges the baseline — consistent with the paper's claim.
  - RFT lowest, as expected: it discards all negative-signal (incorrect)
    samples and has the lowest train_reward (0.25).
  - GRPO_constant / MaxRL sit just below baseline → **inconclusive** per the
    handout's lower-bound logic (untuned lr).
- **Verdict:** matches expectations within single-seed noise; no variant clearly
  beats standard GRPO here.

## 5. Off-policy (32×) — ✅ matches the bias/variance story (⚠ 1 seed)
- **Expectation (p32-38):** naive = biased but no added variance; token-level
  clip = middle of bias/variance; GSPO = sequence-level geometric mean, argued
  to be **more stable** than token-level GRPO. Also: compare clip fractions of
  the two clipping methods.
- **Result (final val_reward):** gspo **0.506** ≈ noclip **0.501** > naive
  **0.487** > clip **0.465**. All ≈ on-policy GRPO (~0.48) → going 32× off-policy
  **kept accuracy while taking 32 steps per inference batch** (the intended
  speedup).
- **Clip fraction (as predicted):** naive & noclip = 0 (no clipping); `clip`
  (ε=0.2) clips **0.8%** of tokens; `gspo` (ε=3e-4 on the geometric-mean
  sequence weight) clips **~24%** (max 48%). GSPO's tiny ε ⇒ high clip fraction,
  exactly as the formulation implies, yet it remains the most stable/best.
- **Verdict:** consistent with theory — gspo ≥ clip (GSPO's stability claim
  holds here), naive survives because at 32× the policy gap is still small.
  The one mild surprise is `clip` being the **weakest** (lower format 0.83);
  with 1 seed and an untuned ε this is within noise.

## Deviations from the spec worth noting
- **Seeds:** 2 (standard) / 1 (everything else) vs. the requested 4.
- **Off-policy `gradient_accumulation_steps`:** our config used **2**; the
  handout (p37) specifies **1** with train_batch_size=8. train_batch_size
  matches; this only affects effective microbatching, not the algorithm.
