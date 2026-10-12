# Results

Numbers come from these auto-generated files:
- `paper/results_tables.md`: the main table, every learner, model and network;
- `paper/cal_summary.md`: calibration;
- `exp3/heur/count_heuristic.md`: the post-hoc shortcut diagnostic;
- `exp5/RESULTS_tables.md` and `exp5/RESULTS_ctrl.md`: the CR axis;
- `exp6/RESULTS_tables.md` and `exp6/mechanism.md`: the In-Context Algebra positive control.

Every run is complete. Two adversarial audits (2026-10-07, 2026-10-09) re-derived the numbers for Experiments 1–5 independently, and this text incorporates their corrections. Experiment 6 was added on 2026-10-11. Its numbers and adapter were checked by a separate adversarial verification pass the same day. Post-hoc and exploratory analyses are labelled where they appear. Deviations from the preregistrations are listed in Methods, "Deviations".

## R1. The fingerprint separates the candidate learners on identical trials (Fig. 1)
The seven preregistered exact learners were scored on the same 1,800 trials that the models see. The preregistered rule returns every learner's own class at n=6 and at n=8, on both accuracy and mean p_gold, with 0 misclassifications.

On associativity-forced (G) cells the four signatures differ sharply (n=6; chance 0.167):

| learner | G accuracy, t=0 → t=4 | held-out t=4 | shape g(t4) − g(t1) | w = held/comp |
|---|---|---|---|---|
| N (notebook) | 0.29 → 0.28 (flat) | 0.29 | — | — |
| N_iso (exemplar matching) | 0.30 → 0.98 (gradual: .30/.69/.85/.98) | 0.30 | +0.35 [+0.26, +0.45] | ≈ 0 |
| A (abstraction) | 0.30 → 1.00 (a step at t=1) | 1.00 | 0.000 | 1.00 |
| B (Bayes, true prior) | 1.00 (flat) | 1.00 | — | — |

The compounding contrast (t=4 minus a length-matched scramble history) is about +0.70 for both A and N_iso. Only held-out transfer separates them: +0.70 for A against +0.03 for N_iso. At n=8, N_iso's G accuracy (.22/.46/.59 at t=0/1/4) tracks the missing mass realised in these stimuli: the target type occurs in the history in 28% (t=1) and 47% (t=4) of trials. It departs from c10's expected-coverage prediction (.38/.69) because the sampled histories over-cover the target type at t=1 (expected 20%) and under-cover it at t=4 (expected 59%).

**Post hoc.** An eighth learner, **C (count shortcut)**, was added after R4's diagnosis. It answers with the target word least represented among the revealed facts. The rule classifies it **R** at n=6 and n=8: acc_L 0.175, G t0 0.226, leak 0.235. This is the correct verdict, because C uses no cell-level, history or axiom information.

## R2. Nine pretrained LMs (ten model–precision configurations) from three families, 0.6B–8B, are recall-only and do not compound (Fig. 2)
Every pretrained configuration is classified **R** at both n=6 and n=8, on both accuracy and p_gold:
- Qwen3-0.6B and Qwen3-0.6B-Base;
- Qwen3-1.7B;
- SmolLM2-1.7B (Llama architecture);
- Qwen3-1.7B-Base (bf16 and nf4†), Qwen3-4B-Base and Qwen3-8B-Base (nf4†);
- OLMo-2-1B (bf16) and OLMo-2-7B (nf4†), a fully open family.

Accuracy on Latin-forced cells at t=0 (acc_L) is 0.14–0.29, and several values are at or below chance (0.167). Every value is below the preregistered threshold of 0.417.

**No compounding.** At n=6, t=4 minus the length-matched scramble history lies between −0.019 and +0.026 (Qwen3-8B-Base: −0.008 [−0.038, +0.022]). Every 95% CI includes 0. Held-out minus scramble lies between −0.04 and +0.02.

Two raw t=0 → t=4 increases were reliable, and both vanish against the scramble history, which shows why the length-matched control is needed:
- OLMo-2-1B at n=6: t4 − t0 = +0.042 [+0.011, +0.073], but t4 − scramble = +0.020 [−0.008, +0.051].
- Qwen3-0.6B at n=8: t4 − t0 = +0.031 [+0.014, +0.049].

**Recall depends on family more than on size.** When the history contains the target's own labelled table (the leak control), G-cell accuracy is:

| model | leak G accuracy |
|---|---|
| Qwen3 models ≤1.7B | 0.52–0.65 (nf4 Qwen3-1.7B-Base: 0.59) |
| Qwen3-4B-Base | 0.95 |
| Qwen3-8B-Base | 0.96 |
| SmolLM2-1.7B | 0.44 |
| OLMo-2-1B | 0.31 |
| OLMo-2-7B | 0.37 |

The larger Qwen3 models retrieve almost perfectly and still do not infer.

**Quantisation (†).** The preregistered control (Qwen3-1.7B-Base, nf4 vs bf16, all 1,800 trials) *failed* its tolerance. nf4 lowered G accuracy in all 17 cells that contain G queries, by 0.020–0.067 (pooled −0.034; mean of per-cell differences −0.040), and per-query argmax agreement was 56%. The class (R) is unchanged, and the compounding contrast is null in both precisions: at n=6, t4 − scramble is −0.007 [−0.037, +0.020] in bf16 and +0.006 [−0.022, +0.035] in nf4.
- What nf4 did *not* do: it did not create a spurious effect, and it preserved strong context use (leak 0.63 → 0.59).
- What this control *cannot* show: whether nf4 would hide a weak real effect at 4B, 7B or 8B, because nothing was there to hide at 1.7B.
- The low bias (pooled −0.034) applies to G accuracy. Pooled L accuracy moved by −0.024 and O by +0.032.

**Answer bias (post hoc; `exp1/diag.py`).** At n=6, Qwen3-1.7B and SmolLM2-1.7B answer G cells with one of the query's own words in 46% and 44% of cases, against 30% for a uniform guess. Over all orders the shares are 39% and 40%, against 26%. Qwen3-0.6B shows no such bias at n=6 (27% vs 30%) or n=8 (14% vs 23%; 19% vs 26% over all orders). At n=5, however, it too exceeds the uniform rate (46% vs 35%; the 1.7B models score 62% and 56%). The correct answer is never a query word (0 of 31,553 G cells).

## R3. Calibration: direct-answer Latin inference is absent up to 8B (Fig. 3)
On the cue sweep (n=6; k = 10 → 28 revealed cells of 36), Latin-forced accuracy pooled over t is (chance 0.167):

| model | k=10 | k=16 | k=22 | k=28 | k28 − k10 [95% CI] |
|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 0.16 | 0.18 | 0.20 | 0.20 | +0.03 [−0.04, +0.10] |
| Qwen3-1.7B | 0.24 | 0.24 | 0.21 | 0.24 | +0.00 [−0.09, +0.09] |
| SmolLM2-1.7B | 0.18 | 0.21 | 0.19 | 0.21 | +0.03 [−0.04, +0.10] |
| Qwen3-1.7B-Base | 0.17 | 0.19 | 0.21 | 0.21 | +0.04 [−0.03, +0.12] |
| Qwen3-4B-Base † | 0.23 | 0.19 | 0.21 | 0.21 | −0.02 [−0.11, +0.06] |
| Qwen3-8B-Base † | 0.18 | 0.22 | 0.25 | 0.26 | +0.08 [+0.01, +0.15] |
| OLMo-2-1B | 0.20 | 0.19 | 0.19 | 0.17 | −0.03 [−0.11, +0.04] |
| OLMo-2-7B † | 0.12 | 0.20 | 0.19 | 0.19 | +0.07 [+0.00, +0.13] |

Two models rise marginally:
- **Qwen3-8B-Base.** The rise appears at both t=0 (0.22 → 0.29) and t=4 (0.14 → 0.23), but neither per-t CI excludes 0. At k=28, accuracy is lower with four history tables (t0 0.29, t4 0.23).
- **OLMo-2-7B.** It starts below chance, at 0.12, and its rise is concentrated at t=0.

Neither comes near the preregistered 0.5 threshold at any k with at least 30 L queries.

**k_L and the conditional k=16 step.**
- At k=6 there are only 4 L queries in 100 trials. Qwen3-0.6B-Base and Qwen3-1.7B-Base each answer 2 of them correctly (0.50).
- Under the 2a rule as frozen, that makes k_L = 6 for these two models. Both are R at k=10, so the conditional k=16 operating point (2d) is triggered.
- The ≥30-query guard that excluded k=6 was added after the Qwen3-0.6B-Base sweep had been seen (Methods, Deviation 4). We therefore also ran 2d as preregistered, on 2026-10-09 after the final audit (`exp2/RESULTS_2d.md`): both remain **R** at k=16. Their acc_L values are 0.177 [0.154, 0.201] and 0.219 [0.196, 0.241] (trial-bootstrap 95% CIs, as computed by the decision rule), below 0.417, and t4 − scramble is −0.034 [−0.092, +0.024] and +0.006 [−0.058, +0.065].
- With the guard, k_L is undefined for all eight calibrated models, so all of them are **non-inferential** on this format. Qwen3-0.6B, the post-trained model, was not swept.

Post hoc (`exp3/heur/count_heuristic.md`): at k=28, 98% of the missing cells are Latin-forced, and one-pass row/column elimination alone leaves a single candidate for 86% of those. Choosing uniformly among the remaining candidates, it would score 0.92 over all missing cells. The direct-answer failure is therefore not about missing information.

**Sorted format (exploratory, added after the ≤1.7B sweeps).** Listing the target's facts row by row helps at k=28:
- Qwen3-4B-Base +0.11 [+0.07, +0.15] and Qwen3-8B-Base +0.10 [+0.06, +0.14], to 0.32 and 0.36;
- smaller gains, with CIs excluding 0, for Qwen3-1.7B-Base (+0.05), OLMo-2-7B (+0.04) and Qwen3-0.6B-Base (+0.03);
- similar but non-significant gains for Qwen3-1.7B (+0.03) and SmolLM2-1.7B (+0.02);
- none for OLMo-2-1B.

Sorted-format gains are therefore not specific to 8B (`paper/cal_summary.md`).

**Instructed header (exploratory).** A one-paragraph task header changed L and G accuracy by −0.083 to +0.042. This covers 32 contrasts: 8 models × k ∈ {10, 22} × {L, G}, pooled over t. Each cell needs ≥30 queries: this guard was added to the k_L rule after the first sweep (Deviation 4), and it is applied here to every class. The mean is −0.006, and 2 of 32 CIs exclude 0, both negative.

**Experiment-2 hypotheses.**
- H1 (L accuracy rises with k for every model) is not supported. Its second part (k_L falls with scale) cannot be tested with the guard; without it, only two of the smallest models have a k_L, an artefact of 4 queries.
- H2 holds vacuously, since no model compounds.
- H3 cannot be tested.

## R4. Transformers trained from scratch converge to a counting shortcut, not to Latin or group inference
Seven GPT-style networks (3.2M parameters at 4 layers; about 0.91B training tokens each, 14,000 steps of up to 65,536 tokens) were trained on the preregistered diets. Every one is classified **R**, and each fails the recall (leak) control (leak G 0.08–0.24).

**Diagnosis (post hoc, `exp3/heur/`).** The learned networks answer each trial with essentially one word: the target word least represented among the revealed facts.
- In A4, 98.6% of predictions fall in that least-revealed set; across cells the set holds about 30% of the target words.
- A4's accuracy equals that of the count shortcut to within 0.001 on all 1,800 main trials (−0.0008 [−0.0017, +0.0001]) and on the cue sweep.
- A1, A2, G4 and I4 match it to within 0.007.

The shortcut also explains the in-training plateau:
- Applied to the in-training evaluation conditions (teacher-forced, using the training generator), the count shortcut scores 0.33 at n=6 and 0.25 at n=8. The networks reach 0.29–0.36 at step 14,000.
- On the identical sequences, A4, G4 and I4 sit within ±0.006 of the shortcut. A1 and A2 sit below it, and A6 sits slightly above it (+0.015 at n=6).
- Cell-level row/column elimination would score 0.70.

It also explains A4's cue-sweep rise. On L cells, A4 goes 0.17 (k=10) → 0.21 → 0.26 → 0.33 (k=28). Over all cells, A4's accuracy matches the shortcut's to within 0.004 at every k: 0.196 vs 0.200 at k=6, 0.214 vs 0.213 at k=10, and 0.334 vs 0.334 at k=28. So the networks did not learn Latin constraints. They learned a global symbol-count statistic, which is a weak consequence of the Latin property.

Among the learned networks, the 6-layer A6 is the only exception, and a small one. It is less constant within a trial (75% of predictions in the least-revealed set), and it beats the shortcut by +0.014 [+0.012, +0.015]. P4, which is not learned, does not follow the shortcut at all: 28% of its predictions fall in the least-revealed set, about the uniform rate, and its accuracy is 0.048 below the shortcut.

**Fingerprint contrasts** (n=6, t=4 minus scramble):

| run | diet | predicted (prereg) | observed | t4 − scr [95% CI] | held-out G t=4 | prediction |
|---|---|---|---|---|---|---|
| A4 | mixed, 4 layers | A (or B-type) | R | −0.002 [−0.026, +0.021] | 0.238 | not met |
| G4 | groups only | B-type or N | R | −0.014 [−0.037, +0.007] | 0.241 | not met |
| I4 | isocopy + Latin | N_iso | R | −0.013 [−0.037, +0.009] | 0.239 | not met |
| A1 | mixed, 1 layer | R or N | R | −0.014 [−0.037, +0.011] | 0.230 | met (trivially: R) |
| A2 | mixed, 2 layers | N-like, O ≫ G | R (n6 t0: O 0.21, G 0.22) | −0.014 [−0.037, +0.011] | 0.226 | not met |
| A6 | mixed, 6 layers | w ≥ A4's; maybe B-type | R (w = 0.57, uninterpretable: comp = −0.22 < 0; A4's w is undefined) | −0.010 [−0.036, +0.015] | 0.239 | not met |
| P4 | fixed pool of 32 tables | R or N | R; *not learned* | +0.013 [−0.010, +0.036] | 0.079 | excluded |

**Preregistered criteria** (`exp3/PREREGISTRATION.md`):
1. **Learning-failure rule.** Every run except P4 exceeds chance + 0.10 (0.267) on in-training n=6 remaining-fact accuracy, with maxima of 0.33–0.36. P4 never does: its maximum is 0.183, and it memorised its pool (training loss 0.005) without transfer to fresh symbols. P4 is therefore *not learned*. It is reported here but excluded from method validation.
   - Post-hoc caveat: the count shortcut alone scores 0.33, so this threshold does not distinguish learning the task from learning the shortcut. A future preregistration should set the threshold relative to learner C.
2. **Method validation: not met.** A4 and I4 are R, against predictions of A/B-type and N_iso. The prereg asks that each mismatch be attributed either to the method or to the training diet. The in-training evaluations, compared post hoc with the count shortcut, attribute it to the training. On the identical training-generator sequences, A4 and I4 stay within ±0.006 of the count shortcut at n=6 and n=8, and A6 exceeds it only slightly (+0.015 at n=6). Their first-fact accuracy after the cue (0.23 for A4 and 0.21 for I4 at step 14,000; chance 0.17) is near the shortcut's level (0.20) and well below row/column elimination (0.30). On the identical trials, the fingerprint returns the same class (R) for the networks as for the exact count learner C.
3. **Depth: not supported.** Held-out G accuracy at t=4 is 0.230 (A1), 0.226 (A2), 0.238 (A4) and 0.239 (A6). This is not monotone, and the CIs overlap. comp is undefined for A1, A2 and A4, because acc_L is below chance. For A6 it is negative: −0.22 [−3.24, +0.16].

## R5. Reasoning probe (exploratory, not preregistered)
Qwen3-1.7B answered Latin-forced cells with thinking on (≤4,096 tokens) and off. The trials were the first 15 t=0 trials each of sorted and raw k=28, and the prompt stated the Latin rule.

| condition | thinking OFF | thinking ON |
|---|---|---|
| sorted k=28 | 4/15 | 1/15 (14/15 truncated) |
| raw k=28 | 3/15 | 1/15 (14/15 truncated) |

- 28 of the 30 thinking runs hit the token budget. Both runs that finished (2,604 and 3,453 tokens) were correct.
- Paired by trial, no trial is correct only with thinking on.
- In an earlier v1 run with a 1,536-token budget, all 13 thinking answers were truncated.
- 6 of the 30 non-thinking answers also reached their 400-token limit without an ANSWER line: sorted trials 6 and 8, raw trials 4, 6, 8 and 12. They were scored by the last candidate word mentioned, and all six were incorrect.

The probe is dominated by truncation. It neither shows nor excludes reasoning-enabled Latin inference at this scale.

## R6. The compute–retrieve (CR) axis does not track retrieval vs inference once format is matched (Experiment 5)
The published CR detector separates its own calibration prompts well in every model (AUC 0.96–0.99 at the chosen layer).

On the fingerprint stimuli, leak_t1 queries are retrievable and base_t1 queries must be inferred. Format, token count and query position are identical across the two. Behaviour differs sharply: accuracy over the queried cells is 0.53–0.97 in leak and 0.17–0.22 in base (0.17–0.21 on the subset of queries projected in Experiment 5).

**H1 (base projects higher, toward "compute", than leak; detector format)** is supported in **0 of 5** models (preregistered criterion ≥3):

| model | base − leak [95% CI] | d | query AUC |
|---|---|---|---|
| Qwen3-0.6B-Base | −27.2 [−32.7, −21.5] | −1.34 | 0.19 |
| Qwen3-1.7B-Base | −158.5 [−173.0, −144.5] | −3.08 | 0.03 |
| Qwen3-1.7B | −0.06 [−1.5, +1.6] | −0.01 | 0.50 |
| Qwen3-4B-Base † | −11.1 [−15.5, −6.5] | −0.67 | 0.34 |
| Qwen3-8B-Base † | +0.70 [−6.7, +8.0] | +0.03 | 0.51 |

- In three models the difference is significant in the **opposite** direction. This outcome lies outside the preregistered interpretations, which covered only "supported" and "not supported".
- H2 (projection rises L < O < G within base) is null in four models and significantly reversed in Qwen3-1.7B-Base (ρ = −0.113 [−0.176, −0.046]).
- In raw format (exploratory), the sign changes.
  - Base > leak, with the CI excluding 0, in 0.6B-Base, 1.7B-Base and 4B. This would meet the ≥3-of-5 criterion had raw format been primary.
  - Base < leak in 1.7B (d = −0.28) and 8B (d = −1.68).
- Projection does not predict correctness within a condition: AUC 0.46–0.55, except 0.38 in Qwen3-1.7B-Base leak.

**Likely surface explanation (Experiment 5b, exploratory, not preregistered).** Leak contexts reuse the target's 6 words in the history, while base histories use 6 other words. Leak contexts therefore have half the vocabulary, about 5× more occurrences of the query words, and a verbatim copy of every target line. The control changes only the history table (100 trials per model, all 26 queries, uncached forwards; `exp5/RESULTS_ctrl.md`):
- **baseRel** keeps the history non-retrievable but writes it in the target's words.
- **leakRel** keeps leak's vocabulary but deranges its answers, so retrievability is the only difference from leak.

Results at the detector layer:
- **Vocabulary explains most of the reversal.** In the three reversed base models (detector format), baseRel reproduces 64–87% of the base–leak gap:
  - Qwen3-0.6B-Base: f_vocab 0.67 [0.54, 0.85];
  - Qwen3-1.7B-Base: 0.87 [0.79, 0.96];
  - Qwen3-4B-Base: 0.64 [0.40, 0.99].
- **With vocabulary fixed (leakRel − leak; predicted > 0), the sign depends on model and wrapper.**
  - Detector format: reversed in 0.6B-Base (d_z −0.36) and 1.7B-Base (−0.28), null in 4B-Base (−0.10), and in the predicted direction in the post-trained 1.7B (+0.36).
  - Raw format: in the predicted direction in all four models (d_z +0.33 to +1.67). At other layers the sign varies.
- **The post-trained null is a cancellation.** In the post-trained model, vocabulary overlap raises the projection (baseRel − base = +1.8) about as much as retrievability lowers it (leak − leakRel = −1.3). This is why the preregistered contrast is null there.

The axis therefore carries an in-context retrieval signal of the predicted sign in two places:
- in the post-trained model, where it is small (d_z +0.36 in detector format), about the size of that model's vocabulary effect;
- in raw format (d_z +0.33 to +1.67). In 1.7B-Base it exceeds the vocabulary effect.

In base models under the detector's chat template, the vocabulary effect (baseRel − base) is 4–12× the retrievability contrast. With vocabulary held fixed, the retrievability contrast itself is reversed in 0.6B-Base and 1.7B-Base and null in 4B-Base (Fig. 5).

## R7. Positive control: a published, task-trained network is classified non-R (Experiment 6; Fig. 6)
In-Context Algebra (Todd et al., ICLR 2026) released a 4-layer, d=1024 transformer trained on in-context cyclic and dihedral group tables.
- **Adapter.** We re-implemented it from the released code and scored the released checkpoint on every cell whose trial fits its 16 symbol tokens.
  - A second, independent forward pass agrees with the adapter on 3,200 of 3,200 answer positions.
  - On fresh samples of its own task it scores 0.97 (shuffled vocabulary) and 0.959 under the upstream evaluation protocol. That is below the 99.4% the authors report, which is the best of about 290 evaluations on one fixed 512-sequence set; released checkpoints score 0.959–0.975 on our fresh sequences.
  - Full-sequence training loss is 0.917, against about 0.93 upstream.
- **Coverage.** Only t ≤ 1 cells fit, so the compounding contrast cannot be measured. Decision-rule steps 1–3 need only the base t=0 cell (`exp6/PREREGISTRATION.md`, frozen before scoring).

**Result:**
- **Not R, not B-type.** At n=6, acc_L is 0.886 [0.820, 0.944], against a threshold of 0.417, and loop-forced cells score 0.91. Associativity-only (G) cells score 0.637 [0.601, 0.675], so g(G t0) = 0.65 [0.59, 0.74] < 0.85.
- **The same holds** at n=8 (acc_L 0.971, G 0.663), at n=5 (0.982, 0.531), on the addendum-2d k=16 stimuli (0.920, 0.704) and on p_gold. In every cell O − G is +0.27 to +0.45.

This is the first real network for which the fingerprint returns a class above R. Its profile is consistent with its authors' mechanistic analysis: identity, cancellation and copying near ceiling, associativity weakest.

**Predictions:**
- Met:
  - not R;
  - leak recall (1.000);
  - G above learner C (+0.41 [+0.37, +0.45], paired);
  - no benefit from a differently labelled history (t1 − t0 on G: −0.025 [−0.071, +0.024]).
- Not met:
  - B-type;
  - lower G on the order-8 types held out from its training (+0.025 [−0.035, +0.085]). Accuracy depends on the specific group more than on whether it was seen: Z2³ scores 0.97 (its smaller relative Z2×Z2 was in training), Q8 0.51.

**Cue sweep.** At t=0 its accuracy over all cells rises from 0.69 (k=6) to 0.998 (k=28). On the same trials Qwen3-8B-Base goes from 0.20 to 0.29, learner C from 0.20 to 0.33, and one-pass row/column elimination from 0.24 to 0.93.

**Exploratory:**
- **Interleaved fact order.** It leaves the same class at n=5, 6 and 8 (n=6: L 0.886, O 0.908, G 0.647).
- **Training trajectory** (6 of 21 released checkpoints, 2 draws).
  - The checkpoints at 0 and 5k are R; those from 10k on are non-R. The upstream training loss drops sharply between those steps.
  - After 20k, Latin and loop accuracy plateau at 0.86–0.91 while g(G t0) keeps rising (0.49 at 10k → 0.65 at 100k).

**Post hoc: mechanism split.**
- The model partly answers by commutative copying. In non-abelian targets it returns a revealed transpose that is wrong 38–46% of the time.
- The right baseline for G cells is a loop-level learner: uniform over the values a loop completion of the cue allows, computed exactly.
- With the transpose hidden, the model beats that learner at n=6 (+0.15 [+0.13, +0.17]) and n=8 (+0.25 [+0.24, +0.27]), so part of its associativity-forced accuracy goes beyond identity and cancellation.
- At n=5 it does not (+0.004 [−0.043, +0.048]).

**What remains untested.** The compounding branch (N vs N_iso vs A) on a real network remains open. It requires a model with more symbol tokens.

## Summary of findings
1. **The fingerprint and decision rule work.** On identical trials, eight exact learners are classified correctly, and abstraction and exemplar matching differ only in held-out transfer, as predicted.
2. **No pretrained model up to 8B shows sparse-cue inference in direct-answer mode, which is the prerequisite for measurable compounding.** No model compounds. All eight calibrated models are non-inferential under the calibration rule with the ≥30-query guard, and the largest show at most a marginal rise with cue density. The preregistered k=16 follow-up for the two models that would qualify without the guard: both remain R at k=16, with acc_L 0.18 and 0.22.
3. **Recall is family-dependent.** Qwen3 rises from 0.52–0.65 at ≤1.7B to 0.95–0.96 at 4–8B. SmolLM2 scores 0.44 and OLMo-2 0.31–0.37.
4. **Small transformers trained from scratch on these diets learn a global counting shortcut,** not Latin, loop or group inference. The fingerprint correctly classifies them with the matching exact learner (R).
5. **A published activation-space "compute vs retrieve" axis does not track retrieval vs inference once format is matched.** Applied as published, it moves significantly in the predicted direction in 0 of 5 models, is significantly reversed in 3, and its sign depends on the prompt wrapper. An exploratory control attributes 64–87% of the reversed gap to context vocabulary. With vocabulary matched, a retrieval signal of the predicted sign appears: small in the post-trained model (d_z 0.36), and d_z 0.33–1.67 without the chat template.
6. **A published, task-trained network is classified above R (positive control).** The In-Context Algebra model infers Latin- and loop-forced cells from sparse cues (0.89–0.99) but associativity-only cells only partly (0.53–0.70; non-R, non-B). Its class changes during training (exploratory). The compounding branch on a real network remains untested.
