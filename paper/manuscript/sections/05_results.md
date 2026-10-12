## Results {#sec:results}

Every run is complete. All numbers below come from auto-generated result tables, and the full table for every model, network and idealised learner except Asl is in @app:main. Two adversarial audits re-derived the Experiment 1–5 numbers that existed at the time, and later verification passes (2026-10-09 to 2026-10-11) re-derived the results produced after them and those of Experiment 6; the text incorporates all their corrections. Post hoc and exploratory analyses are labelled where they appear, and every deviation from the preregistrations is listed in @app:deviations.

### The fingerprint separates the idealised learners {#sec:r-ideal}

We scored the seven preregistered exact learners (@sec:learners) on the same 1,800 trials that the models see. The preregistered decision rule (@sec:rule) returns every learner's own class at n=6 and at n=8, on both accuracy and mean $p_{\mathrm{gold}}$, with 0 misclassifications; Asl, which infers "group" from an identity alone, is classified A. @tab:learners lists the signatures at n=6.

Table: Idealised learners on the identical trials (n=6; accuracy on associativity-forced cells, chance 0.167). Class is the rule's verdict at n=6 and at n=8. t4 − scr is G accuracy at t=4 minus the length-matched scramble history; shape is g(G t4) − g(G t1); $w$ = held/comp is given only where comp is clearly above 0. Learner C was added post hoc. {#tab:learners}

| learner | class | G, t=0 → t=4 | held-out G, t=4 | t4 − scr [95% CI] | shape [95% CI] | $w$ |
|---|---|---|---|---|---|---|
| R (recall) | R | 0.179 → 0.172 | 0.168 | −0.004 [−0.032, +0.026] | — | — |
| N (notebook) | N | 0.292 → 0.281 | 0.287 | −0.007 [−0.047, +0.036] | −0.013 [−0.059, +0.036] | — |
| N_iso (exemplar) | N_iso | 0.299 → 0.981 | 0.298 | +0.718 [+0.682, +0.751] | +0.351 [+0.257, +0.446] | ≈ 0 |
| A (abstraction) | A | 0.301 → 1.000 | 1.000 | +0.701 [+0.675, +0.727] | 0.000 | 1.00 |
| B (true prior) | B-type | 1.000 → 1.000 | 1.000 | 0.000 | — | — |
| Bh (hierarchical) | B-type | 1.000 → 1.000 | 1.000 | +0.601 [+0.574, +0.630] | — | — |
| C (count; post hoc) | R | 0.226 → 0.216 | 0.244 | −0.016 [−0.041, +0.009] | — | — |

On associativity-forced (G) cells the four core signatures differ sharply (@fig:ideal). The notebook learner N stays flat, at 0.29 at t=0, 0.28 at t=4 and 0.29 on held-out types. Exemplar matching (N_iso) rises gradually, through 0.30, 0.69, 0.85 and 0.98 at t = 0, 1, 2 and 4, but stays at 0.30 on held-out types; its shape statistic is +0.35 [+0.26, +0.45] and its $w$ is about 0. Abstraction (A) steps to 1.00 at t=1 and keeps that level on held-out types, with a shape of 0.000 and $w$ = 1.00. Bayes with the true prior (B) is at 1.00 from t=0. B and Bh are classified B-type at step 3, before the compounding contrast is consulted.

The compounding contrast, t=4 minus a length-matched scramble history, is about +0.70 for both A and N_iso. Only held-out transfer separates them cleanly: held-out minus scramble is +0.70 for A against +0.03 for N_iso. A compounding contrast on its own therefore cannot tell abstraction from exemplar reuse, and the held-out types are what make the distinction.

At n=8, N_iso's G accuracy (0.22, 0.46 and 0.59 at t = 0, 1 and 4) tracks the missing mass realised in these stimuli: the target type occurs in the history in 28% of trials at t=1 and 47% at t=4. It departs from the design-stage expected-coverage prediction (0.38 and 0.69) because the sampled histories over-cover the target type at t=1 (expected 20%) and under-cover it at t=4 (expected 59%).

**Post hoc.** We added an eighth learner, C (count shortcut), after the diagnosis reported in @sec:r-trained. It answers with the target word least represented among the revealed facts. The rule classifies it R at n=6 and at n=8, with $\mathrm{acc}_L$ 0.175, G t0 0.226 and leak 0.235. This is the correct verdict, because C uses no cell-level, history or axiom information.

![Accuracy of the exact idealised learners on associativity-forced (G) cells against the number of earlier complete tables t, at n=6 (left) and n=8 (right), with held-out types at t=4 shown as diamonds. Abstraction (A) and exemplar matching (N_iso) both rise with t, but only A transfers to held-out types; the notebook learner (N) stays flat, and Bayes with the true prior (B) is at ceiling from t=0.](fig/fig1_ideal_learners.png){#fig:ideal}

### Pretrained models are recall-only and do not compound {#sec:r-llm}

All nine checkpoints (ten model–precision configurations) from three families are classified R at both n=6 and n=8, on both accuracy and $p_{\mathrm{gold}}$ (@tab:recall). The set comprises the post-trained Qwen3-0.6B and Qwen3-1.7B; Qwen3-0.6B-Base, Qwen3-1.7B-Base (bf16 and nf4†), Qwen3-4B-Base and Qwen3-8B-Base (nf4†) [@qwen3]; SmolLM2-1.7B, which has a Llama architecture [@smollm2]; and OLMo-2-1B (bf16) and OLMo-2-7B (nf4†), a fully open family [@olmo2]. Accuracy on Latin-forced cells at t=0 ($\mathrm{acc}_L$) is 0.14–0.29. Several values are at or below chance (0.167), and every value is below the preregistered threshold of 0.417.

Table: Pretrained models on the main trials (n=6). Every configuration is class R at n=6 and at n=8. $\mathrm{acc}_L$ is L-cell accuracy at base t=0 (R threshold 0.417); t4 − scr is G accuracy at t=4 minus the scramble history; leak G is G accuracy when the history contains the target's own labelled table. † nf4, for which the preregistered quantisation control failed. {#tab:recall}

| model | precision | $\mathrm{acc}_L$ | t4 − scr [95% CI] | leak G |
|---|---|---|---|---|
| Qwen3-0.6B | bf16 | 0.289 | −0.019 [−0.053, +0.014] | 0.620 |
| Qwen3-1.7B | bf16 | 0.193 | +0.012 [−0.016, +0.040] | 0.648 |
| Qwen3-0.6B-Base | bf16 | 0.219 | −0.003 [−0.035, +0.030] | 0.515 |
| Qwen3-1.7B-Base | bf16 | 0.175 | −0.007 [−0.037, +0.020] | 0.627 |
| Qwen3-1.7B-Base | nf4 † | 0.140 | +0.006 [−0.022, +0.035] | 0.590 |
| Qwen3-4B-Base | nf4 † | 0.202 | +0.026 [−0.005, +0.056] | 0.949 |
| Qwen3-8B-Base | nf4 † | 0.219 | −0.008 [−0.038, +0.022] | 0.958 |
| SmolLM2-1.7B | bf16 | 0.184 | +0.002 [−0.031, +0.035] | 0.442 |
| OLMo-2-1B | bf16 | 0.228 | +0.020 [−0.008, +0.051] | 0.307 |
| OLMo-2-7B | nf4 † | 0.158 | −0.011 [−0.038, +0.017] | 0.365 |

#### No compounding

At n=6, t4 − scramble lies between −0.019 and +0.026 (Qwen3-8B-Base: −0.008 [−0.038, +0.022]), and every 95% CI includes 0 (@fig:llm). Held-out minus scramble lies between −0.04 and +0.02.

Two raw increases from t=0 to t=4 were reliable, and both vanish against the scramble history. OLMo-2-1B at n=6 gains +0.042 [+0.011, +0.073] from t=0 to t=4, but only +0.020 [−0.008, +0.051] relative to scramble. Qwen3-0.6B at n=8 gains +0.031 [+0.014, +0.048], and +0.008 relative to scramble, with a CI that includes 0. Neither raw gain survives the length-matched control, so both are consistent with effects of context length alone, which is why that control is needed.

#### Recall depends on family more than on size

When the history contains the target's own labelled table (the leak control), G-cell accuracy is 0.52–0.65 for the Qwen3 models up to 1.7B (0.59 for nf4 Qwen3-1.7B-Base), 0.95 for Qwen3-4B-Base and 0.96 for Qwen3-8B-Base. It is 0.44 for SmolLM2-1.7B, and 0.31 and 0.37 for OLMo-2-1B and OLMo-2-7B. The larger Qwen3 models thus retrieve almost perfectly and still do not infer.

#### Quantisation control (†)

The preregistered control (Qwen3-1.7B-Base, nf4 against bf16, all 1,800 trials) *failed* its tolerance. nf4 lowered G accuracy in all 17 cells that contain G queries, by 0.020–0.067 (pooled −0.034; mean of per-cell differences −0.040), and per-query argmax agreement was 56%. The class (R) is unchanged, and the compounding contrast is null in both precisions: at n=6, t4 − scramble is −0.007 [−0.037, +0.020] in bf16 and +0.006 [−0.022, +0.035] in nf4.

nf4 therefore did not create a spurious effect, and it preserved strong context use (leak 0.63 → 0.59). What this control cannot show is whether nf4 would hide a weak real effect at 4B, 7B or 8B, because at 1.7B there was nothing to hide. The low bias (pooled −0.034) applies to G accuracy; pooled L accuracy moved by −0.024 and O accuracy by +0.032.

#### Answer bias (post hoc)

A post-hoc diagnostic (`exp1/diag.py`) shows that at n=6 Qwen3-1.7B and SmolLM2-1.7B answer G cells with one of the query's own words in 46% and 44% of cases, against 30% for a uniform guess; over all orders the shares are 39% and 40%, against 26%. Qwen3-0.6B shows no such bias at n=6 (27% against 30%) or at n=8 (14% against 23%; 19% against 26% over all orders). At n=5, however, it too exceeds the uniform rate (46% against 35%), and the two 1.7B models score 62% and 56%. Because the correct answer is never a query word (0 of 31,553 G cells), this bias pulls G accuracy below chance.

![Compounding contrasts on G cells at n=6: t=4 minus the length-matched scramble history for seen types (left) and for held-out types (right), with 95% trial-bootstrap CIs, for nine pretrained checkpoints and three idealised learners. Every pretrained interval on seen types includes 0, whereas A and N_iso sit near +0.70 and only A keeps that gain on held-out types.](fig/fig2_llm_contrasts.png){#fig:llm}

### Calibration: no direct-answer Latin inference up to 8B {#sec:r-cal}

On the cue sweep (n=6; k = 10 → 28 revealed cells of 36), Latin-forced accuracy pooled over t stays close to chance (0.167) for every pretrained model (@tab:cal; @fig:cal). No model comes near the preregistered threshold of 0.5 at any k with at least 30 L queries.

Table: Latin-forced (L) accuracy on the cue sweep (n=6, pooled over t=0 and t=4; chance 0.167), with the change from k=10 to k=28 and its 95% trial-bootstrap CI. The preregistered $k_L$ threshold is 0.5. † nf4. A4 is the trained four-layer network of @sec:r-trained. {#tab:cal}

| model | k=10 | k=16 | k=22 | k=28 | k28 − k10 [95% CI] |
|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 0.163 | 0.177 | 0.200 | 0.196 | +0.034 [−0.036, +0.102] |
| Qwen3-1.7B | 0.236 | 0.238 | 0.213 | 0.240 | +0.004 [−0.089, +0.087] |
| SmolLM2-1.7B | 0.179 | 0.205 | 0.189 | 0.209 | +0.030 [−0.041, +0.100] |
| Qwen3-1.7B-Base | 0.171 | 0.194 | 0.208 | 0.213 | +0.042 [−0.034, +0.115] |
| Qwen3-4B-Base † | 0.228 | 0.194 | 0.211 | 0.207 | −0.021 [−0.106, +0.062] |
| Qwen3-8B-Base † | 0.179 | 0.219 | 0.254 | 0.263 | +0.084 [+0.010, +0.153] |
| OLMo-2-1B | 0.203 | 0.190 | 0.191 | 0.168 | −0.035 [−0.112, +0.038] |
| OLMo-2-7B † | 0.122 | 0.202 | 0.191 | 0.191 | +0.069 [+0.004, +0.128] |
| A4 (trained) | 0.171 | 0.209 | 0.265 | 0.330 | +0.160 [+0.099, +0.217] |

Two pretrained models rise marginally. For Qwen3-8B-Base the rise appears at both t=0 (0.22 → 0.29) and t=4 (0.14 → 0.23), but neither per-t CI excludes 0, and at k=28 accuracy is lower with four history tables than with none (0.23 against 0.29). OLMo-2-7B starts below chance, at 0.12, and its rise is concentrated at t=0.

#### The Latin threshold and the conditional k=16 step

At k=6 there are only 4 L queries in 100 trials, and Qwen3-0.6B-Base and Qwen3-1.7B-Base each answer 2 of them correctly (0.50). Under the addendum-2a rule as frozen, that makes $k_L$ = 6 for these two models. Both are R at k=10, so the conditional k=16 operating point (addendum 2d) is triggered. The guard of at least 30 L queries, which excluded k=6, was added after the Qwen3-0.6B-Base sweep had been seen (@app:deviations, row 4). We therefore also ran 2d as preregistered, on 2026-10-09, after the final audit. Both models remain R at k=16. Their $\mathrm{acc}_L$ values are 0.177 [0.154, 0.201] and 0.219 [0.196, 0.241] (trial-bootstrap 95% CIs, as computed by the decision rule), below 0.417, and t4 − scramble is −0.034 [−0.092, +0.024] and +0.006 [−0.058, +0.065].

With the guard, $k_L$ is undefined for all eight calibrated models, so all of them are *non-inferential* on this format. Qwen3-0.6B, the post-trained model, was not swept.

#### Missing information does not explain the failure (post hoc)

At k=28, 98% of the missing cells are Latin-forced (784 of 800), and one-pass row/column elimination alone leaves a single candidate for 86% of those (675 of 784; `exp2/k28_elimination.md`). Choosing uniformly among the remaining candidates, this elimination would score 0.92 over all missing cells. The direct-answer failure is therefore not about missing information.

#### Sorted format (exploratory)

Listing the target's facts row by row, a variant added after the sweeps of the first three models (Qwen3-0.6B-Base, Qwen3-1.7B and SmolLM2-1.7B; @app:deviations), helps at k=28. Qwen3-4B-Base gains +0.108 [+0.066, +0.149] and Qwen3-8B-Base +0.101 [+0.061, +0.144], reaching 0.32 and 0.36. Smaller gains with CIs excluding 0 appear for Qwen3-1.7B-Base (+0.046), OLMo-2-7B (+0.045) and Qwen3-0.6B-Base (+0.029). Qwen3-1.7B (+0.027) and SmolLM2-1.7B (+0.024) show similar but non-significant gains, and OLMo-2-1B shows none (+0.005). Sorted-format gains are therefore not specific to 8B (@app:calib).

#### Instructed header (exploratory)

A one-paragraph task header changed L and G accuracy by −0.083 to +0.042 over 32 contrasts: 8 models × k ∈ {10, 22} × {L, G}, pooled over t. Each contrast requires at least 30 queries in its cell. This guard was added to the $k_L$ rule after the first sweep (@app:deviations, row 4), and it is applied here to every class. The mean change is −0.006, and 2 of the 32 CIs exclude 0, both negative.

#### Experiment-2 hypotheses

The preregistered H1, that L accuracy rises with k for every model, is not supported. Its second part, that $k_L$ falls with scale, cannot be tested with the guard; without it, only two of the smallest models have a $k_L$, an artefact of 4 queries. H2, that any compounding appears only in the larger models, holds vacuously, since no model compounds. H3, that held-out transfer separates A from N_iso in a model that compounds, cannot be tested.

![Latin-forced (L) accuracy on the cue sweep (n=6, pooled over t=0 and t=4) against the number of revealed cells k for the eight calibrated pretrained models and the trained network A4, with chance and the preregistered $k_L$ threshold of 0.5 marked. No pretrained model approaches the threshold, and A4's rise follows the count shortcut.](fig/fig3_calibration.png){#fig:cal}

### Trained transformers learn a counting shortcut {#sec:r-trained}

We trained seven GPT-style networks from scratch on the preregistered diets (@sec:trained), with 3.2M parameters at 4 layers and about 0.91B training tokens each (14,000 steps of up to 65,536 tokens). Every one is classified R, and every one also fails the recall (leak) control, with leak G accuracy of 0.08–0.24 (@tab:app-main-class). For these networks, class R means only that $\mathrm{acc}_L$ is below threshold: they neither recall nor infer at sparse cues.

#### Diagnosis (post hoc)

The learned networks answer each trial with essentially one word: the target word least represented among the revealed facts (`exp3/heur/`). In A4, 98.6% of predictions fall in that least-revealed set, which across cells holds about 30% of the target words. A4's accuracy equals that of the count shortcut C to within 0.001 on all 1,800 main trials (−0.0008 [−0.0017, +0.0001]) and on the cue sweep, and A1, A2, G4 and I4 match C to within 0.007.

The shortcut also explains the in-training plateau. Applied to the in-training evaluation conditions (teacher-forced, with the training generator), the count shortcut scores 0.33 at n=6 and 0.25 at n=8, and the networks reach 0.29–0.36 at step 14,000. On the identical sequences, averaged over the evaluation conditions of each order, A4, G4 and I4 sit within ±0.006 of the shortcut, A1 and A2 sit below it, and A6 sits slightly above it (+0.015 at n=6). Cell-level row/column elimination would score 0.70.

It also explains A4's rise on the cue sweep. On L cells A4 goes from 0.17 at k=10 through 0.21 and 0.26 to 0.33 at k=28 (@tab:cal). Over all cells its accuracy matches the shortcut's to within 0.004 at every k: 0.196 against 0.200 at k=6, 0.214 against 0.213 at k=10, and 0.334 against 0.334 at k=28. The networks therefore did not learn Latin constraints. They learned a global symbol-count statistic, which is a weak consequence of the Latin property.

Among the learned networks, the 6-layer A6 is the only exception, and a small one. It is less constant within a trial (75% of its predictions fall in the least-revealed set), and it beats the shortcut by +0.014 [+0.012, +0.015]. P4, which is not learned, does not follow the shortcut at all: 28% of its predictions fall in the least-revealed set, about the uniform rate, and its accuracy is 0.048 below the shortcut's.

Table: Trained networks: preregistered predictions against observed classes (n=6). t4 − scr is G accuracy at t=4 minus the scramble history; held-out G is G accuracy on held-out types at t=4 (chance 0.167). P4 is *not learned* under the preregistered learning-failure rule and is excluded from method validation. {#tab:trained}

| run | diet | predicted | observed | t4 − scr [95% CI] | held-out G | prediction |
|---|---|---|---|---|---|---|
| A4 | mixed, 4 layers | A (or B-type) | R | −0.002 [−0.026, +0.021] | 0.238 | not met |
| G4 | groups only | B-type or N | R | −0.014 [−0.037, +0.007] | 0.241 | not met |
| I4 | isocopy + Latin | N_iso | R | −0.013 [−0.037, +0.009] | 0.239 | not met |
| A1 | mixed, 1 layer | R or N | R | −0.014 [−0.037, +0.011] | 0.230 | met (trivially) |
| A2 | mixed, 2 layers | N-like, O ≫ G | R | −0.014 [−0.037, +0.011] | 0.226 | not met |
| A6 | mixed, 6 layers | $w$ ≥ A4's; maybe B-type | R | −0.010 [−0.036, +0.015] | 0.239 | not met |
| P4 | fixed pool of 32 tables | R or N | R; not learned | +0.013 [−0.010, +0.036] | 0.079 | excluded |

Two entries need a note. A2 was predicted to be N-like with O ≫ G, but at n=6, t=0 its O and G accuracies are 0.21 and 0.22. A6 has $w$ = 0.57, but the value is uninterpretable because its comp is negative (−0.22), and A4's $w$ is undefined.

#### Learning-failure rule (preregistered)

We applied the three preregistered Experiment-3 criteria (`exp3/PREREGISTRATION.md`) after the second audit. Under the learning-failure rule, every run except P4 exceeds chance + 0.10 (0.267) on in-training n=6 remaining-fact accuracy, with maxima of 0.33–0.36. P4 never does: its maximum is 0.183, and it memorised its pool (training loss 0.005) without transfer to fresh symbols. P4 is therefore *not learned*; we report it but exclude it from method validation. A post-hoc caveat applies: the count shortcut alone scores 0.33, so this threshold does not distinguish learning the task from learning the shortcut. A future preregistration should set the threshold relative to learner C.

#### Method validation: not met

A4 and I4 are R, against predictions of A or B-type and of N_iso. The preregistration asks that each mismatch be attributed either to the method or to the training diet. The in-training evaluations, compared post hoc with the count shortcut, attribute it to training. On the identical training-generator sequences, A4 and I4 stay, on average over conditions, within ±0.006 of the count shortcut at n=6 and n=8. Their first-fact accuracy after the cue (0.23 for A4 and 0.21 for I4 at step 14,000; chance 0.17) is near the shortcut's level (0.20) and well below row/column elimination (0.30). On the identical trials, the fingerprint returns the same class (R) for the networks as for the exact count learner C.

#### Depth: not supported

Held-out G accuracy at t=4 is 0.230 (A1), 0.226 (A2), 0.238 (A4) and 0.239 (A6). This is not monotone in depth, and the CIs overlap. comp is undefined for A1, A2 and A4, because their $\mathrm{acc}_L$ is below chance; for A6 it is negative, −0.22 [−3.24, +0.16].

![G-cell accuracy (n=6) against the number of earlier tables t for the seven networks trained from scratch, beside the exact learners A, N_iso and C, with held-out types at t=4 shown at right. Every learned network lies close to the count-shortcut learner C, while P4, which was not learned, sits below chance.](fig/fig4_trained_transformers.png){#fig:trained}

### Reasoning probe (exploratory) {#sec:r-probe}

This probe was exploratory and not preregistered (@sec:probe). Qwen3-1.7B answered Latin-forced cells with thinking on (up to 4,096 tokens) and off. The trials were the first 15 t=0 trials each of the sorted and raw k=28 formats, and the prompt stated the Latin rule. With thinking off, the model answered 4 of 15 sorted and 3 of 15 raw trials correctly. With thinking on, it answered 1 of 15 correctly in each format, and 14 of the 15 runs in each format were truncated.

In all, 28 of the 30 thinking runs hit the token budget. Both runs that finished (2,604 and 3,453 tokens) were correct, but paired by trial, no trial is correct only with thinking on. In an earlier run with a 1,536-token budget, all 13 thinking answers were truncated. Six of the 30 non-thinking answers also reached their 400-token limit without an ANSWER line (sorted trials 6 and 8; raw trials 4, 6, 8 and 12). They were scored by the last candidate word mentioned, and all six were incorrect.

The probe is dominated by truncation. It neither shows nor excludes reasoning-enabled Latin inference at this scale.

### The compute–retrieve axis under matched format {#sec:r-cr}

The compute–retrieve detector [@ramdan2026cr] separates its own calibration prompts well in every model we tested, with AUC 0.96–0.99 at the chosen layer (@sec:cr). On the fingerprint stimuli, queries in the leak condition at t=1 are retrievable, and queries in the base condition at t=1 must be inferred. Format, token count and query position are identical across the two. Behaviour differs sharply: accuracy over the queried cells is 0.53–0.97 in leak and 0.17–0.22 in base (0.17–0.21 on the subset of queries projected in Experiment 5).

The preregistered H1, that base projects higher than leak (toward "compute") in the detector format, is supported in 0 of 5 models, against a criterion of at least 3 (@tab:cr). In three models the difference is significant in the *opposite* direction. This outcome lies outside the preregistered interpretations, which covered only "supported" and "not supported". H2, that projection rises from L to O to G cells within base, is null in four models and significantly reversed in Qwen3-1.7B-Base ($\rho$ = −0.113 [−0.176, −0.046]). In an exploratory analysis, projection also does not predict correctness within a condition (AUC 0.46–0.55, except 0.38 in Qwen3-1.7B-Base leak).

In raw format, without the chat template (exploratory), the sign changes. Base exceeds leak, with the CI excluding 0, in Qwen3-0.6B-Base, Qwen3-1.7B-Base and Qwen3-4B-Base, which would meet the 3-of-5 criterion had raw format been primary. Base falls below leak in Qwen3-1.7B (d = −0.28) and Qwen3-8B-Base (d = −1.68).

Table: The CR axis on the fingerprint stimuli at the detector layer. Experiment 5 (preregistered H1, detector format): base − leak projection with 95% trial-bootstrap CI, Cohen's d and query-level AUC (base > leak); raw-format d is exploratory. Experiment 5b (exploratory, detector format): $f_{\mathrm{vocab}}$ is the share of the base–leak gap reproduced by vocabulary alone, and leakRel − leak (predicted > 0) is the paired effect with vocabulary fixed. † nf4. {#tab:cr}

| model | base − leak [95% CI] | d | AUC | raw d | $f_{\mathrm{vocab}}$ [95% CI] | leakRel − leak $d_z$ |
|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | −27.2 [−32.7, −21.5] | −1.34 | 0.19 | +0.79 | 0.67 [0.54, 0.85] | −0.36 |
| Qwen3-1.7B-Base | −158.5 [−173.0, −144.5] | −3.08 | 0.03 | +0.44 | 0.87 [0.79, 0.96] | −0.28 |
| Qwen3-1.7B | −0.06 [−1.5, +1.6] | −0.01 | 0.50 | −0.28 | n/a (no gap) | +0.36 |
| Qwen3-4B-Base † | −11.1 [−15.5, −6.5] | −0.67 | 0.34 | +0.77 | 0.64 [0.40, 0.99] | −0.10 |
| Qwen3-8B-Base † | +0.70 [−6.7, +8.0] | +0.03 | 0.51 | −1.68 | not run | not run |

#### Context vocabulary (Experiment 5b; exploratory, not preregistered)

Leak and base are matched on length and format, but not on vocabulary. Leak contexts reuse the target's 6 words in the history, while base histories use 6 other words. Leak contexts therefore have half the vocabulary, about 5× more occurrences of the query words, and a verbatim copy of every target line. The control changes only the history table (100 trials per model, all 26 queries, uncached forwards; @app:cr). In **baseRel** the history stays non-retrievable but is written in the target's words. In **leakRel** the history keeps leak's vocabulary but its answers are deranged, so retrievability is the only difference from leak.

Vocabulary explains most of the reversal. In the three base models reversed in detector format, baseRel reproduces 64–87% of the base–leak gap: $f_{\mathrm{vocab}}$ is 0.67 [0.54, 0.85] for Qwen3-0.6B-Base, 0.87 [0.79, 0.96] for Qwen3-1.7B-Base and 0.64 [0.40, 0.99] for Qwen3-4B-Base (@fig:cr).

With vocabulary fixed (leakRel − leak, predicted > 0), the sign depends on model and wrapper. In detector format the contrast is reversed in Qwen3-0.6B-Base ($d_z$ −0.36) and Qwen3-1.7B-Base (−0.28), null in Qwen3-4B-Base (−0.10), and in the predicted direction only in the post-trained Qwen3-1.7B (+0.36). In raw format it is in the predicted direction in all four models ($d_z$ +0.33 to +1.67), although at other layers the sign varies.

The post-trained null is a cancellation. In Qwen3-1.7B, vocabulary overlap raises the projection (baseRel − base = +1.8) about as much as retrievability lowers it (leak − leakRel = −1.3), which is why the preregistered contrast is null there.

The axis therefore carries an in-context retrieval signal of the predicted sign in two places. In the post-trained model it is small ($d_z$ +0.36 in detector format), about the size of that model's vocabulary effect. In raw format it ranges from $d_z$ +0.33 to +1.67, and in Qwen3-1.7B-Base it exceeds the vocabulary effect. In the base models under the detector's chat template, by contrast, the vocabulary effect (baseRel − base) is 4–12× the retrievability contrast, and with vocabulary held fixed the retrievability contrast itself is reversed in Qwen3-0.6B-Base and Qwen3-1.7B-Base and null in Qwen3-4B-Base.

![Experiment 5b (exploratory): CR projection for each condition minus leak, in units of the pooled SD of the leak and base trial means, at the detector layer and in detector (chat-template) format, for four Qwen3 models (8B was not run). Writing a non-retrievable history in the target's words (baseRel) reproduces most of the base–leak gap, and making the retrievable history inconsistent (leakRel) lowers the projection in Qwen3-0.6B-Base and Qwen3-1.7B-Base, against the prediction, has no reliable effect in Qwen3-4B-Base, and raises it, as predicted, only in the post-trained Qwen3-1.7B.](fig/fig5_cr_axis_control.png){#fig:cr}

### Positive control: the In-Context Algebra model {#sec:r-ica}

The In-Context Algebra study [@todd2026ica] released a 4-layer, d=1024 transformer trained on in-context cyclic and dihedral group tables. We re-implemented it from the released code and scored the released checkpoint on every cell whose trial fits its 16 symbol tokens (@sec:ica). A second, independent forward pass agrees with our adapter on 3,200 of 3,200 answer positions. On fresh samples of its own task the model scores 0.97 with shuffled vocabulary and 0.959 under the upstream evaluation protocol. The latter is below the 99.4% the authors report, which is the best of about 290 evaluations on one fixed 512-sequence set; on our fresh sequences, the released checkpoints from 20k steps on score 0.959–0.975. Full-sequence training loss is 0.917, against about 0.93 upstream (@app:ica).

Only cells with t ≤ 1 fit the model's vocabulary, so the compounding contrast cannot be measured. Decision-rule steps 1–3 need only the base t=0 cell, and this primary analysis was frozen before scoring (`exp6/PREREGISTRATION.md`).

Table: The released In-Context Algebra model under decision-rule steps 1–3 (base, t=0, accuracy). The R threshold is 1/n + 0.25, and g(G t0) ≥ 0.85 would make the model B-type. The class is the same on mean $p_{\mathrm{gold}}$. {#tab:ica}

| cells | $\mathrm{acc}_L$ [95% CI] | R threshold | O | G [95% CI] | g(G t0) [95% CI] | class |
|---|---|---|---|---|---|---|
| n=6, k=10 | 0.886 [0.820, 0.944] | 0.417 | 0.910 | 0.637 [0.601, 0.675] | 0.65 [0.59, 0.74] | non-R, non-B |
| n=8, k=16 | 0.971 [0.909, 1.000] | 0.375 | 0.988 | 0.663 [0.631, 0.699] | 0.64 [0.58, 0.71] | non-R, non-B |
| n=5, k=6 | 0.982 [0.952, 1.000] | 0.450 | 0.895 | 0.531 [0.485, 0.574] | 0.42 [0.36, 0.48] | non-R, non-B |
| n=6, k=16 (2d) | 0.920 [0.900, 0.938] | 0.417 | 0.920 | 0.704 [0.651, 0.758] | 0.71 [0.64, 0.79] | non-R, non-B |

The model is neither R nor B-type. At n=6, $\mathrm{acc}_L$ is 0.886 [0.820, 0.944], against a threshold of 0.417, and loop-forced cells score 0.91. Associativity-only (G) cells score 0.637 [0.601, 0.675], so g(G t0) = 0.65 [0.59, 0.74], below the B-type threshold of 0.85. The same holds at n=8 ($\mathrm{acc}_L$ 0.971, G 0.663), at n=5 (0.982 and 0.531), on the addendum-2d k=16 stimuli (0.920 and 0.704) and on $p_{\mathrm{gold}}$ (@tab:ica; @fig:ica). In every non-leak cell of the main design (base $t = 0$ at $n = 5$, 6 and 8; base and held-out $t = 1$ at $n = 6$ and 8), O − G is +0.27 to +0.45; on the addendum-2d $k = 16$ stimuli the gap is smaller, +0.20 to +0.22.

This is the first real network for which the fingerprint returns a class above R. Its profile is consistent with its authors' mechanistic analysis: identity, cancellation and copying are near ceiling, and associativity is weakest.

Four of the preregistered predictions are met: the model is not R; leak recall is 1.000; G accuracy exceeds learner C (+0.411 [+0.369, +0.453], paired); and a differently labelled history brings no benefit (t1 − t0 on G: −0.025 [−0.071, +0.024]). Two are *not met*. The model is not B-type, and G accuracy is not lower on the order-8 types held out from its training (+0.025 [−0.035, +0.085]). Accuracy depends on the specific group more than on whether it was seen: Z2³ scores 0.97 (its smaller relative Z2×Z2 was in training), and Q8 scores 0.51.

On the t=0 cue sweep, the model's accuracy over all cells rises from 0.69 at k=6 to 0.998 at k=28. On the same trials, Qwen3-8B-Base goes from 0.20 to 0.29, learner C from 0.20 to 0.33, and one-pass row/column elimination from 0.24 to 0.93.

#### Fact order and training trajectory (exploratory)

Interleaving all context facts in one random order leaves the class unchanged at n=5, 6 and 8 (n=6: L 0.886, O 0.908, G 0.647). We also scored 6 of the 21 released training checkpoints (2 symbol draws). The checkpoints at 0 and 5k steps are R, and those from 10k on are non-R; the upstream training loss drops sharply between those steps. After 20k, Latin and loop accuracy plateau at 0.86–0.91, while g(G t0) keeps rising, from 0.49 at 10k to 0.65 at 100k. The class therefore changes during training.

#### Mechanism split (post hoc)

The model partly answers by commutative copying. In non-abelian targets, when the transposed cell is revealed but differs from the queried cell, the model returns the transpose in 38% (n=6) to 46% (n=8) of cases. The appropriate baseline for G cells is a loop-level learner, which chooses uniformly among the values a loop completion of the cue allows, computed exactly. G cells are by definition not loop-forced, so beating this learner requires group structure beyond identity and cancellation. With the transpose hidden, the model beats it at n=6 (+0.148 [+0.130, +0.166]) and n=8 (+0.254 [+0.236, +0.273]), so part of its associativity-forced accuracy goes beyond identity and cancellation. At n=5 it does not (+0.004 [−0.043, +0.048]).

What remains untested is the compounding branch (N against N_iso against A) on a real network. Testing it requires a model with more symbol tokens.

![(a) Accuracy on L-, O- and G-forced cells at n=6, base t=0, for the released In-Context Algebra model, Qwen3-8B-Base, the trained network A4 and the idealised learners N, C and B. (b) Accuracy over all queried cells on the t=0 cue sweep for the In-Context Algebra model and Qwen3-8B-Base, against learner C and one-pass row/column elimination.](fig/fig6_in_context_algebra.png){#fig:ica}
