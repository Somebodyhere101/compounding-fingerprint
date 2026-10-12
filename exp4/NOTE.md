# Experiment 4: reasoning probe (exploratory; added 2026-10-07 ~17:20 after Experiments 1–3 showed no sparse-cue inference)

**Question.** Is the chance-level Latin inference of ≤4B models a limit of direct (single-token) answering rather than of capability?

**Design.** Qwen3-1.7B (post-trained) answers one Latin-forced cell per trial, from Experiment 2's t=0 trials:
- sorted k=28, sorted k=22 and raw k=28 (L cells);
- raw k=16 (G cells).

There are 30 trials per condition. The chat prompt states that the table is a Latin square. Each prompt is answered twice: thinking ON (≤1,536 tokens) and thinking OFF (≤400 tokens). Decoding is greedy.

**Outcome.** Accuracy, thinking ON minus OFF, per condition. Truncated thinking counts as incorrect and is reported separately.

**Not preregistered.** Because the prompt names the Latin rule, this probe measures *inference given the rule*, not induction of the rule.

## Amendment (2026-10-08 20:52)
In v1 (budget 1,536 thinking tokens), all 13 thinking-mode answers on sorted k=28 L cells were truncated before an answer, so it was stopped. Non-thinking scored 4/14. Raw records are kept in `res/_v1_trunc1536.jsonl`.

v2 changes only the budget and the sample:
- 4,096 thinking tokens;
- 15 trials each of `kso_n6_k28` and `ks_n6_k28` (L cells);
- thinking on and off.

It runs after Experiment 5. Truncations are reported, not dropped.

## Outcome (v2, finished 2026-10-09 16:37; `res/qwen3-1.7b_cot_v2.jsonl`, 60 records)
| condition | thinking OFF | thinking ON |
|---|---|---|
| sorted k=28 (`kso_n6_k28`) | 4/15 | 1/15 (14/15 truncated) |
| raw k=28 (`ks_n6_k28`) | 3/15 | 1/15 (14/15 truncated) |

- 28 of the 30 thinking runs hit the 4,096-token budget. Both runs that finished (2,604 and 3,453 tokens) were correct.
- Paired by trial, no trial is correct only with thinking ON, and 5 are correct only with thinking OFF.
- The v2 non-thinking answers on trials 0–13 reproduce v1 exactly (14/14).
- 6 of the 30 non-thinking answers (sorted trials 6 and 8; raw trials 4, 6, 8 and 12) also reached the 400-token limit without an ANSWER line. cot.py scored them by the last target word mentioned, and all six were incorrect.

**Reading:** the probe is dominated by truncation. It neither shows nor excludes reasoning-enabled Latin inference at 1.7B. A conclusive version needs a larger thinking budget, or a model that reasons more concisely.
