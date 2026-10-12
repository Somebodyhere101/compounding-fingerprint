# Experiment 3: preregistration (controlled from-scratch transformers)

Frozen 2026-10-07 before any full training run. The code, tests, throughput and run grid are in `README.md`. That file's "Proposed minimal run grid" table is the authoritative statement of configurations and predictions; its key points are restated here.

## Purpose
1. **Method validation with known training diets.** Does the Experiment-1 fingerprint and decision rule (`../exp1/analyze.py`, unchanged) recover the class implied by what a network could have learned?
2. **A test of the bounded-inference predictions about depth.**

## Fixed settings
- d_model 256, 8 heads, RoPE.
- 14,000 steps of 65,536 tokens, warmup 500, cosine decay to 10%.
- lr 6e-4 (1e-3 for 1–2 layers).
- Orders {4, 5, 6, 7, 8, 10}.
- Held-out iso types (never in training for any `noheld` run): S3 (n=6), D4 and Q8 (n=8).
- Evaluation: the exact Experiment-1 stimuli, scored with `eval_fp.py --draws 4` (four random symbol assignments per trial), then analysed with Experiment 1's preregistered rule.

## Runs and predicted classes (n=6 primary)
| run | training diet | predicted class | key signature |
|---|---|---|---|
| A4 | mixed environments (group 0.5, Latin 0.25, loop 0.25), 4 layers | **A** (alternative: **B-type** if the cue alone identifies the environment) | step at t=1; held-out ≈ seen gain (w ≈ 1); shape ≈ 0 |
| G4 | groups only, 4 layers | **B-type or N**: no compounding (comp < 0.15) | history uninformative ⇒ no compounding |
| P4 | groups only, fixed pool of 32 labelled tables per order | **R or N** | no transfer to fresh symbols |
| I4 | isomorphic-copy + Latin (never a group or identity) | **N_iso** | gradual (missing-mass) rise, w ≈ 0, high leak |
| A1 | as A4, 1 layer | **R or N** | no compounding |
| A2 | as A4, 2 layers | **N-like, with O ≫ G** | identity/cancellation without associativity |
| A6 | as A4, 6 layers | held-out w ≥ A4's; may become **B-type** (comp shrinks) | bounded inference relaxes with depth |

## Decision criteria for the paper's claims
- **The method is validated** if A4 is classified A or B-type, I4 is classified N_iso, and P4/A1 are classified R or N. Each mismatch is reported as a failure of the method or of the training diet, and diagnosed with the in-training evaluations (`runs/*/eval.csv`).
- **The depth claim** is supported if comp or held-out G accuracy is non-decreasing from A1 → A2 → A4. A6 is reported either way.
- **Learning-failure rule.** A run whose in-training remaining-fact accuracy at n=6 never exceeds chance + 0.10 is declared *not learned* and excluded from the method-validation claims. It is still reported.

## Allowed adaptations (logged if used)
- If A4 is *not learned* by 14k steps, A4 and the dependent runs may be retrained once with 2× steps. No other hyperparameter search.
- If P4 generalises (in-training fresh-symbol accuracy rises), it is rerun once with pool size M = 8, as the README pre-specifies.
