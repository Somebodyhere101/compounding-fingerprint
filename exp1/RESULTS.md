# Experiment 1 results: compounding fingerprint, small pretrained LMs

Run 2026-10-06/07 on an RTX 5070 Laptop GPU (8 GB). There were 1,800 trials and 57,200 scored cells per model. The decision rule and thresholds were frozen in advance in [PREREGISTRATION.md](PREREGISTRATION.md). Full tables are in [RESULTS_tables.md](RESULTS_tables.md), produced by `analyze.py`; response biases come from `diag.py`.

## Headline
All three models are classified **R (recall only)** under the preregistered rule, at both n=6 and n=8. They show no structural inference beyond chance and no compounding from earlier tables. They can still copy: when the history contains the target's own labelled table (the leak control), accuracy jumps to 0.44–0.65.

## Key numbers (raw accuracy on associativity-forced "G" cells; chance 0.167 at n=6, 0.125 at n=8)

| model | n | G at t=0 | G at t=4 | t=4 minus t=0 [95% CI] | t=4 minus scramble [95% CI] | held-out t=4 | leak t=1 | class |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 6 | 0.150 | 0.160 | +0.009 [−0.021, +0.040] | −0.019 [−0.053, +0.014] | 0.139 | 0.620 | R |
| Qwen3-0.6B | 8 | 0.132 | 0.163 | +0.031 [+0.014, +0.049] | +0.008 [−0.008, +0.025] | 0.139 | — | R |
| Qwen3-1.7B | 6 | 0.115 | 0.120 | +0.005 [−0.022, +0.033] | +0.012 [−0.016, +0.040] | 0.091 | 0.648 | R |
| Qwen3-1.7B | 8 | 0.109 | 0.115 | +0.006 [−0.012, +0.023] | +0.010 [−0.007, +0.027] | 0.098 | — | R |
| SmolLM2-1.7B (Llama arch.) | 6 | 0.159 | 0.144 | −0.016 [−0.046, +0.017] | +0.002 [−0.031, +0.035] | 0.147 | 0.442 | R |
| SmolLM2-1.7B (Llama arch.) | 8 | 0.117 | 0.100 | −0.017 [−0.032, −0.002] | −0.001 [−0.017, +0.014] | 0.110 | — | R |

Latin-forced accuracy without history (acc_L) is 0.18–0.29 against a chance of 0.17. That is below the preregistered R threshold of 1/n + 0.25 for every model.

## Interpretation
1. **No compounding at ≤1.7B.** Two t=0→t=4 changes have CIs that exclude zero, and neither is a compounding effect:
   - The only reliable *increase* (Qwen3-0.6B, n=8, +0.031) is matched by the length-matched scramble control (+0.008 vs scramble, n.s.). It is a context-length effect, not structure.
   - SmolLM2 at n=8 shows a small reliable *decrease* (−0.017 [−0.032, −0.002]). It is also n.s. against scramble (−0.001 [−0.017, +0.014]).
2. **A copy bias in two of the three models.** On G cells the gold answer is never one of the two query words (0 of 31,553 G cells; corrected 2026-10-09, after the earlier "33 of 36,058" was found not to match the stimuli). A uniform guess names a query word in about 30% of n=6 G cells. At n=6 these models answer with one of the query's own words in the following shares; Qwen3-0.6B is below the uniform rate at n=6 and n=8 (but not at n=5: 46% vs 35%), so only the two 1.7B models show the bias at n ≥ 6:
   - Qwen3-1.7B: 46% (62% at n=5);
   - SmolLM2: 44% (56% at n=5);
   - Qwen3-0.6B: 27% (46% at n=5).

   This produces below-chance G accuracy. *(Corrected 2026-10-07 after the audit. An earlier version reported 78% and 50–68%. Those figures double-counted diagonal cells, where both query words are the same, and mixed n=5 into the n=6 numbers. `diag.py` now reports the union as COPY_ANY.)*
3. **Recall works.** The leak control rules out a broken pipeline: the models retrieve a table they have seen, and fail only when inference is needed.
4. **What this does and doesn't show.** It shows that models at this scale sit at the bottom of the fingerprint ladder (R), so the compounding question cannot be asked of them. It says nothing yet about 4–8B models, where the paper's question lives.

## Caveats
- Batched scoring in bf16 changes about 2–6% of near-tie argmax decisions compared with sequential scoring. The maximum change in p_gold is 0.05. This noise is unbiased across conditions, and every contrast is reported on both accuracy and mean p_gold with the same conclusions.
- There are few L cells at n=8 (15–34 per cell), so acc_L at n=8 is noisy. The n=6 rule is primary.
- 100 trials per cell, the c10 minimum.
