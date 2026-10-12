## Full results for the learners, models and networks {#app:main}

@tab:app-main-class and @tab:app-main-g give the complete main table: every idealised learner except Asl (which the rule classifies A at $n = 6$ and $n = 8$; @sec:r-ideal), and every pretrained model and trained network, scored on the identical 1,800 trials. Classes come from the unchanged preregistered decision rule, applied both to accuracy and to mean $p_{\mathrm{gold}}$; the R threshold at $n = 6$ is $\mathrm{acc}_L < 0.417$, and chance on G cells is 0.167. Contrasts carry trial-bootstrap 95% CIs (2,000 resamples, with seeds fixed per test so that the intervals are reproducible).

Three caveats apply to these tables. First, † marks nf4: the preregistered quantisation control failed its tolerance (maximum per-cell G difference 0.067; @app:impl), so absolute G accuracy is biased low by about 0.04 and per-query predictions are unstable, although at 1.7B the class and the t4 − scramble contrast are unchanged. Second, learner C was added post hoc (2026-10-09). Third, for the trained networks class R means only that $\mathrm{acc}_L$ is below threshold: every network also fails the leak control, so it neither recalls nor infers at sparse cues. P4 is *not learned* under the Experiment-3 learning-failure rule and is excluded from method validation.

The preregistered derived quantities comp, held and $w$ are meaningful only above class R, because the rescaling g is undefined when $\mathrm{acc}_L$ is near chance. For the idealised learners at $n = 6$ they separate as designed: N_iso has comp +0.819 [+0.772, +0.861] and held −0.002 [−0.043, +0.039] ($w \approx 0$), A has comp and held both +0.839 [+0.802, +0.856] ($w$ = 1.00), and N has comp −0.013 [−0.064, +0.023]. For every class-R model and network these quantities are uninterpretable, and we do not report them.

Table: Class, Latin-forced accuracy and leak-control accuracy for every learner (except Asl), model and network on the identical 1,800 trials. Class is given on accuracy / mean $p_{\mathrm{gold}}$ at $n = 6$ and on accuracy at $n = 8$; $\mathrm{acc}_L$ is at $n = 6$, $t = 0$; leak G is G-cell accuracy in the leak control. † nf4. {#tab:app-main-class}

| group | learner/model | precision | class n6 (acc / $p_{\mathrm{gold}}$) | class n8 | $\mathrm{acc}_L$ | leak G |
|---|---|---|---|---|---|---|
| idealised | R (recall) | exact | R / R | R | 0.132 | 1.000 |
| idealised | N (Latin notebook) | exact | N / N | N | 1.000 | 1.000 |
| idealised | N_iso (exemplar) | exact | N_iso / N_iso | N_iso | 1.000 | 1.000 |
| idealised | A (abstraction) | exact | A / A | A | 1.000 | 1.000 |
| idealised | B (true prior) | exact | B-type / B-type | B-type | 1.000 | 1.000 |
| idealised | Bh (hierarchical) | exact | B-type / B-type | B-type | 1.000 | 1.000 |
| idealised | C (count shortcut; post hoc) | exact | R / R | R | 0.175 | 0.235 |
| pretrained | Qwen3-0.6B | bf16 | R / R | R | 0.289 | 0.620 |
| pretrained | Qwen3-0.6B-Base | bf16 | R / R | R | 0.219 | 0.515 |
| pretrained | OLMo-2-1B | bf16 | R / R | R | 0.228 | 0.307 |
| pretrained | Qwen3-1.7B | bf16 | R / R | R | 0.193 | 0.648 |
| pretrained | Qwen3-1.7B-Base | bf16 | R / R | R | 0.175 | 0.627 |
| pretrained | Qwen3-1.7B-Base | nf4 † | R / R | R | 0.140 | 0.590 |
| pretrained | SmolLM2-1.7B | bf16 | R / R | R | 0.184 | 0.442 |
| pretrained | Qwen3-4B-Base | nf4 † | R / R | R | 0.202 | 0.949 |
| pretrained | OLMo-2-7B | nf4 † | R / R | R | 0.158 | 0.365 |
| pretrained | Qwen3-8B-Base | nf4 † | R / R | R | 0.219 | 0.958 |
| trained | A1 | fp32/bf16 | R / R | R | 0.123 | 0.232 |
| trained | A2 | fp32/bf16 | R / R | R | 0.158 | 0.225 |
| trained | A4 | fp32/bf16 | R / R | R | 0.140 | 0.225 |
| trained | A6 | fp32/bf16 | R / R | R | 0.219 | 0.229 |
| trained | G4 | fp32/bf16 | R / R | R | 0.114 | 0.213 |
| trained | P4 | fp32/bf16 | R / R | R | 0.184 | 0.079 |
| trained | I4 | fp32/bf16 | R / R | R | 0.140 | 0.235 |

Table: G-cell accuracy at $n = 6$ by history condition (base $t = 0$, base $t = 4$, scramble $t = 4$, held-out $t = 4$) and the compounding contrast, base $t = 4$ minus scramble $t = 4$, with its trial-bootstrap 95% CI. Chance is 0.167. † nf4. {#tab:app-main-g}

| learner/model | G t0 | G t4 | G scr4 | G held4 | t4 − scr [95% CI] |
|---|---|---|---|---|---|
| R (recall) | 0.179 | 0.172 | 0.175 | 0.168 | −0.004 [−0.032, +0.026] |
| N (Latin notebook) | 0.292 | 0.281 | 0.288 | 0.287 | −0.007 [−0.047, +0.036] |
| N_iso (exemplar) | 0.299 | 0.981 | 0.263 | 0.298 | +0.718 [+0.682, +0.751] |
| A (abstraction) | 0.301 | 1.000 | 0.299 | 1.000 | +0.701 [+0.675, +0.727] |
| B (true prior) | 1.000 | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] |
| Bh (hierarchical) | 1.000 | 1.000 | 0.399 | 1.000 | +0.601 [+0.574, +0.630] |
| C (count shortcut; post hoc) | 0.226 | 0.216 | 0.232 | 0.244 | −0.016 [−0.041, +0.009] |
| Qwen3-0.6B | 0.150 | 0.160 | 0.179 | 0.139 | −0.019 [−0.053, +0.014] |
| Qwen3-0.6B-Base | 0.179 | 0.187 | 0.190 | 0.163 | −0.003 [−0.035, +0.030] |
| OLMo-2-1B | 0.169 | 0.211 | 0.191 | 0.209 | +0.020 [−0.008, +0.051] |
| Qwen3-1.7B | 0.115 | 0.120 | 0.108 | 0.091 | +0.012 [−0.016, +0.040] |
| Qwen3-1.7B-Base (bf16) | 0.178 | 0.150 | 0.157 | 0.164 | −0.007 [−0.037, +0.020] |
| Qwen3-1.7B-Base (nf4 †) | 0.157 | 0.122 | 0.116 | 0.100 | +0.006 [−0.022, +0.035] |
| SmolLM2-1.7B | 0.159 | 0.144 | 0.142 | 0.147 | +0.002 [−0.031, +0.035] |
| Qwen3-4B-Base † | 0.225 | 0.207 | 0.182 | 0.187 | +0.026 [−0.005, +0.056] |
| OLMo-2-7B † | 0.173 | 0.166 | 0.176 | 0.161 | −0.011 [−0.038, +0.017] |
| Qwen3-8B-Base † | 0.214 | 0.209 | 0.217 | 0.215 | −0.008 [−0.038, +0.022] |
| A1 | 0.220 | 0.224 | 0.238 | 0.230 | −0.014 [−0.037, +0.011] |
| A2 | 0.219 | 0.213 | 0.227 | 0.226 | −0.014 [−0.037, +0.011] |
| A4 | 0.228 | 0.219 | 0.221 | 0.238 | −0.002 [−0.026, +0.021] |
| A6 | 0.246 | 0.234 | 0.244 | 0.239 | −0.010 [−0.036, +0.015] |
| G4 | 0.231 | 0.217 | 0.231 | 0.241 | −0.014 [−0.037, +0.007] |
| P4 | 0.096 | 0.088 | 0.075 | 0.079 | +0.013 [−0.010, +0.036] |
| I4 | 0.241 | 0.215 | 0.228 | 0.239 | −0.013 [−0.037, +0.009] |

#### Exemplar matching at $n = 8$
At $n = 8$, N_iso's G accuracy (0.22, 0.46 and 0.59 at $t$ = 0, 1 and 4) tracks the missing mass realised in these stimuli: the target type occurs in the history in 28% of trials at $t = 1$ and 47% at $t = 4$. It departs from the design-stage expected-coverage prediction (0.38 and 0.69) because the sampled histories over-cover the target type at $t = 1$ (expected 20%) and under-cover it at $t = 4$ (expected 59%).

#### Answer bias (post hoc)
A post-hoc diagnostic (`exp1/diag.py`) shows that at $n = 6$ Qwen3-1.7B and SmolLM2-1.7B answer G cells with one of the query's own words in 46% and 44% of cases, against 30% for a uniform guess; over all orders the shares are 39% and 40%, against 26%. Qwen3-0.6B shows no such bias at $n = 6$ (27% against 30%) or at $n = 8$ (14% against 23%; 19% against 26% over all orders). At $n = 5$, however, it too exceeds the uniform rate (46% against 35%), and the two 1.7B models score 62% and 56%. Because the correct answer is never a query word (0 of 31,553 G cells), this bias pulls G accuracy below chance.

## Held-out types against type-matched baselines {#app:typematched}

This check is post hoc and was not preregistered. Held-out targets are non-abelian types only (S3 at $n = 6$; D4 or Q8 at $n = 8$), with abelian histories, whereas the base and scramble cells mix target types. We therefore restricted base $t = 0$, base $t = 4$ and scramble $t = 4$ to the trials whose target type is a held-out type (at $n = 6$, 50, 46 and 43 trials; at $n = 8$, 47, 39 and 40) and compared them with the 100 held-out trials of each order, using pooled G-cell accuracy and trial-bootstrap 95% CIs with 2,000 resamples. @tab:app-typematched gives the $n = 6$ results: G-cell accuracy on S3 targets at base $t = 0$, scramble $t = 4$ and held-out $t = 4$; held-out minus type-matched scramble; type-matched base $t = 4$ minus held-out, the gain from having the target's type in the history; and, for comparison, the raw held-out contrast against all scramble trials as computed in the main analysis (summarised as a range in @sec:r-llm).

Table: Type-matched held-out check at $n = 6$ (post hoc), with 95% CIs. † nf4; other pretrained models bf16, trained networks fp32/bf16. {#tab:app-typematched}

| learner/model | G t0 | G scr4 | G held4 | held − scr, matched [95% CI] | t4 − held, matched [95% CI] | held − scr, all [95% CI] |
|---|---|---|---|---|---|---|
| R (recall) | 0.177 | 0.184 | 0.168 | −0.016 [−0.053, +0.021] | +0.017 [−0.019, +0.054] | −0.008 [−0.037, +0.021] |
| N (Latin notebook) | 0.299 | 0.280 | 0.287 | +0.007 [−0.041, +0.057] | +0.008 [−0.039, +0.056] | −0.000 [−0.039, +0.040] |
| N_iso (exemplar) | 0.307 | 0.264 | 0.298 | +0.033 [−0.005, +0.073] | +0.681 [+0.628, +0.721] | +0.034 [−0.000, +0.066] |
| A (abstraction) | 0.286 | 0.305 | 1.000 | +0.695 [+0.654, +0.735] | +0.000 [+0.000, +0.000] | +0.701 [+0.675, +0.726] |
| B (true prior) | 1.000 | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| Bh (hierarchical) | 1.000 | 0.458 | 1.000 | +0.542 [+0.504, +0.580] | +0.000 [+0.000, +0.000] | +0.601 [+0.574, +0.630] |
| C (count; post hoc) | 0.215 | 0.243 | 0.244 | +0.000 [−0.029, +0.032] | −0.025 [−0.058, +0.007] | +0.012 [−0.014, +0.037] |
| Qwen3-0.6B | 0.147 | 0.138 | 0.139 | +0.001 [−0.037, +0.038] | −0.005 [−0.041, +0.034] | −0.040 [−0.073, −0.007] |
| Qwen3-0.6B-Base | 0.185 | 0.201 | 0.163 | −0.038 [−0.077, +0.002] | +0.028 [−0.014, +0.073] | −0.027 [−0.056, +0.003] |
| OLMo-2-1B | 0.182 | 0.197 | 0.209 | +0.012 [−0.025, +0.048] | +0.001 [−0.035, +0.035] | +0.018 [−0.012, +0.048] |
| Qwen3-1.7B | 0.109 | 0.090 | 0.091 | +0.001 [−0.031, +0.030] | +0.021 [−0.010, +0.052] | −0.017 [−0.046, +0.010] |
| Qwen3-1.7B-Base | 0.174 | 0.172 | 0.164 | −0.008 [−0.050, +0.029] | −0.018 [−0.053, +0.016] | +0.007 [−0.019, +0.036] |
| Qwen3-1.7B-Base † | 0.157 | 0.134 | 0.100 | −0.034 [−0.072, +0.004] | +0.033 [−0.001, +0.069] | −0.016 [−0.044, +0.011] |
| SmolLM2-1.7B | 0.153 | 0.165 | 0.147 | −0.018 [−0.068, +0.026] | −0.001 [−0.036, +0.037] | +0.005 [−0.027, +0.037] |
| Qwen3-4B-Base † | 0.226 | 0.188 | 0.187 | −0.001 [−0.031, +0.033] | +0.032 [−0.007, +0.069] | +0.005 [−0.022, +0.034] |
| OLMo-2-7B † | 0.169 | 0.184 | 0.161 | −0.023 [−0.057, +0.012] | +0.013 [−0.019, +0.046] | −0.015 [−0.040, +0.011] |
| Qwen3-8B-Base † | 0.209 | 0.222 | 0.215 | −0.007 [−0.048, +0.031] | +0.011 [−0.023, +0.048] | −0.002 [−0.031, +0.027] |
| A1 | 0.220 | 0.245 | 0.230 | −0.015 [−0.045, +0.014] | +0.007 [−0.025, +0.039] | −0.008 [−0.033, +0.017] |
| A2 | 0.228 | 0.238 | 0.226 | −0.011 [−0.041, +0.018] | −0.007 [−0.042, +0.027] | −0.001 [−0.026, +0.025] |
| A4 | 0.231 | 0.236 | 0.238 | +0.003 [−0.024, +0.029] | −0.010 [−0.040, +0.020] | +0.017 [−0.004, +0.042] |
| A6 | 0.253 | 0.259 | 0.239 | −0.020 [−0.047, +0.007] | +0.014 [−0.020, +0.046] | −0.005 [−0.029, +0.020] |
| G4 | 0.233 | 0.243 | 0.241 | −0.003 [−0.030, +0.023] | −0.004 [−0.030, +0.024] | +0.009 [−0.014, +0.032] |
| P4 | 0.100 | 0.088 | 0.079 | −0.009 [−0.039, +0.022] | −0.001 [−0.027, +0.023] | +0.005 [−0.019, +0.028] |
| I4 | 0.241 | 0.230 | 0.239 | +0.009 [−0.017, +0.036] | −0.011 [−0.040, +0.018] | +0.010 [−0.011, +0.033] |

The separation of the idealised learners survives the restriction. A keeps its held-out gain (+0.695 [+0.654, +0.735]); N_iso's held-out contrast stays small (+0.033 [−0.005, +0.073]), and on the same target types it gains +0.681 [+0.628, +0.721] when the target's type is in the history. For every pretrained model, both type-matched contrasts include 0. Qwen3-0.6B's raw held-out contrast, the only one of the ten that excluded 0 (−0.040 [−0.073, −0.007]), is +0.001 [−0.037, +0.038] type-matched, so it reflected the difficulty of S3 targets rather than the history.

At $n = 8$ the pattern is the same: held-out minus type-matched scramble is +0.774 [+0.751, +0.797] for A, and N_iso gains +0.326 [+0.201, +0.453] when the target's type is in the history. For every language model and trained network, the type-matched held-out minus scramble and base $t = 4$ minus held-out contrasts at $n = 8$ include 0, except P4's held-out minus scramble (−0.020 [−0.036, −0.004]).

Held-out histories also move accuracy relative to no history on the same target types in a few configurations. At $n = 6$ they lower S3 accuracy in two nf4 configurations, Qwen3-1.7B-Base (held − t0 −0.056 [−0.090, −0.022]) and Qwen3-4B-Base (−0.039 [−0.077, −0.001]). At $n = 8$, held − t0 excludes 0 for Qwen3-0.6B (+0.021 [+0.004, +0.038]), OLMo-2-1B (+0.027 [+0.008, +0.045]) and OLMo-2-7B (−0.019 [−0.037, −0.003]), and its upper bound rounds to −0.000 for Qwen3-1.7B-Base in nf4 (−0.019 [−0.037, −0.000]) and A2 (−0.009 [−0.017, −0.000]). Held-out histories never contain the target's type, so these differences are not transfer of the target's structure; they reflect either non-specific effects of having a history or differences between the trial sets, since the history-blind learner R shows one of the same size at $n = 8$ (held − t0 −0.023 [−0.043, −0.003]).

## Multiple comparisons {#app:mult}

The main analyses apply no correction for multiple comparisons. This check is post hoc and was not preregistered. It recomputes the raw G-accuracy contrasts of the ten pretrained-model configurations with a trial bootstrap of 20,000 resamples and applies a Bonferroni correction over the 20 model-by-order $t = 4$ minus $t = 0$ contrasts (99.75% intervals, $p < 0.0025$) and over the 10 $n = 6$ $t = 4$ minus scramble nulls (99.5% intervals) (@tab:app-mult). With more resamples, the 95% intervals can differ in the third decimal from the 2,000-resample intervals of the main text.

Table: Multiple-comparisons check for the pretrained models (post hoc; 20,000 resamples): raw $t = 4$ minus $t = 0$ G-accuracy contrasts with 95% CIs and bootstrap $p$ at $n = 6$ and $n = 8$, and the $n = 6$ $t = 4$ minus scramble contrast with its Bonferroni-adjusted 99.5% interval. † nf4; other models bf16. {#tab:app-mult}

| model | n=6 t4 − t0 [95% CI] | p | n=8 t4 − t0 [95% CI] | p | n=6 t4 − scr [99.5% CI] |
|---|---|---|---|---|---|
| Qwen3-0.6B | +0.009 [−0.021, +0.039] | 0.5435 | +0.031 [+0.014, +0.049] | 0.0003 | −0.019 [−0.066, +0.029] |
| Qwen3-0.6B-Base | +0.008 [−0.026, +0.043] | 0.6332 | +0.006 [−0.010, +0.022] | 0.4694 | −0.003 [−0.050, +0.044] |
| OLMo-2-1B | +0.042 [+0.010, +0.074] | 0.0103 | +0.013 [−0.002, +0.027] | 0.0783 | +0.020 [−0.022, +0.062] |
| Qwen3-1.7B | +0.005 [−0.024, +0.034] | 0.7463 | +0.006 [−0.011, +0.023] | 0.4870 | +0.012 [−0.029, +0.053] |
| Qwen3-1.7B-Base | −0.028 [−0.058, +0.002] | 0.0659 | −0.005 [−0.019, +0.009] | 0.5042 | −0.007 [−0.050, +0.033] |
| Qwen3-1.7B-Base † | −0.035 [−0.065, −0.005] | 0.0202 | −0.014 [−0.029, +0.001] | 0.0601 | +0.006 [−0.036, +0.046] |
| SmolLM2-1.7B | −0.016 [−0.048, +0.017] | 0.3362 | −0.017 [−0.032, −0.002] | 0.0254 | +0.002 [−0.045, +0.048] |
| Qwen3-4B-Base † | −0.018 [−0.049, +0.014] | 0.2759 | −0.022 [−0.040, −0.004] | 0.0180 | +0.026 [−0.017, +0.069] |
| OLMo-2-7B † | −0.007 [−0.036, +0.022] | 0.6350 | −0.021 [−0.035, −0.007] | 0.0034 | −0.011 [−0.048, +0.028] |
| Qwen3-8B-Base † | −0.004 [−0.036, +0.028] | 0.7932 | −0.008 [−0.022, +0.005] | 0.2253 | −0.008 [−0.050, +0.035] |

Of the 20 $t = 4$ minus $t = 0$ contrasts, 6 have uncorrected $p < 0.05$ (2 increases, 4 decreases). Only Qwen3-0.6B at $n = 8$ (+0.031; 99.75% interval [+0.006, +0.058]) survives Bonferroni; OLMo-2-1B's +0.042 at $n = 6$ (99.75% interval [−0.008, +0.090]) does not. Every Bonferroni-adjusted interval of the $t = 4$ minus scramble nulls includes 0, and the largest upper bound rises from +0.056 to +0.069 (Qwen3-4B-Base).

## Calibration and the addendum-2d operating point {#app:calib}

The cue sweep of Experiment 2a used $n = 6$, histories of $t = 0$ and $t = 4$ tables, and cues from $k = 6$ to $k = 28$ revealed cells, with 50 trials per combination of $t$ and $k$. @tab:app-cal-k gives Latin-forced accuracy by $k$, pooled over $t$, for the eight calibrated language models and for the trained network A4; $k = 6$ is omitted because its 100 trials contain only 4 L queries. Two pretrained models rise marginally from $k = 10$ to $k = 28$, with CIs excluding 0 (Qwen3-8B-Base and OLMo-2-7B), and neither comes near the preregistered 0.5 threshold at any $k$ with at least 30 L queries. For Qwen3-8B-Base the rise appears at both $t = 0$ (0.22 → 0.29) and $t = 4$ (0.14 → 0.23), but neither per-$t$ CI excludes 0, and at $k = 28$ accuracy is lower with four history tables than with none (0.23 against 0.29). OLMo-2-7B starts below chance, at 0.12, and its rise is concentrated at $t = 0$. A4 rises more (+0.160), but its accuracy tracks the count shortcut at every $k$ (@app:trained).

Table: Latin-forced accuracy by cue size $k$ ($n = 6$, pooled over $t = 0$ and $t = 4$; chance 0.167), with the $k = 28$ minus $k = 10$ difference and its 95% CI. † nf4. {#tab:app-cal-k}

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
| A4 (trained, 4 layers) | 0.171 | 0.209 | 0.265 | 0.330 | +0.160 [+0.099, +0.217] |

![Latin-forced (L) accuracy on the cue sweep (n=6, pooled over t=0 and t=4) against the number of revealed cells k for the eight calibrated pretrained models and the trained network A4, with chance and the preregistered $k_L$ threshold of 0.5 marked. No pretrained model approaches the threshold, and A4's rise follows the count shortcut.](fig/fig3_calibration.png){#fig:cal}

#### The Latin threshold and the conditional $k = 16$ step
The guard requiring at least 30 pooled L queries per $k$ was added after the Qwen3-0.6B-Base sweep had been seen, because $k = 6$ has only 4 L cells in 100 trials (@app:deviations, row 4). Under the 2a rule as frozen, without the guard, Qwen3-0.6B-Base and Qwen3-1.7B-Base each answer 2 of the 4 L queries at $k = 6$ correctly (0.50), so $k_L = 6$; being R at $k = 10$, they trigger the conditional $k = 16$ operating point of addendum 2d. This is the only adaptive step allowed: it regenerates the $n = 6$ main contrasts at $k = 16$ (100 trials per cell, new seeds) and takes $\mathrm{acc}_L$ from $k = 16$ base $t = 0$. We ran it as specified on 2026-10-09, after the final audit (600 trials per model; chance 0.167, R threshold $\mathrm{acc}_L < 0.417$), and report both readings. Both models remain R, and neither gains beyond the scramble control (@tab:app-cal-2d). The $\mathrm{acc}_L$ intervals are trial-bootstrap percentile intervals, as computed by the decision rule. With the guard, $k_L$ is undefined for all eight calibrated models.

Table: Addendum 2d: the $k = 16$ operating point at $n = 6$ for the two models that the frozen rule sends there. G columns give G-cell accuracy at base $t = 0$, base $t = 4$, and scramble / held-out $t = 4$. {#tab:app-cal-2d}

| model | class (acc / $p_{\mathrm{gold}}$) | $\mathrm{acc}_L$ ($k = 16$, $t = 0$) [95% CI] | G t0 | G t4 | G scr4 / held4 | t4 − scr [95% CI] |
|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | R / R | 0.177 [0.154, 0.201] | 0.160 | 0.168 | 0.202 / 0.204 | −0.034 [−0.092, +0.024] |
| Qwen3-1.7B-Base | R / R | 0.219 [0.196, 0.241] | 0.152 | 0.182 | 0.176 / 0.211 | +0.006 [−0.058, +0.065] |

#### Format variants (exploratory)
Neither of the two exploratory variants is used for classification. The sorted variant lists the target's facts grouped by row, at $k \in$ {16, 22, 28}; it was added after the first three models swept, all at or below 1.7B (Qwen3-0.6B-Base, Qwen3-1.7B and SmolLM2-1.7B), showed chance-level L accuracy (deviation 5 in @app:deviations). @tab:app-cal-sorted gives the paired sorted-minus-raw difference on L cells. At $k = 28$ sorting helps Qwen3-4B-Base (+0.108 [+0.066, +0.149]) and Qwen3-8B-Base (+0.101 [+0.061, +0.144]) most, which reach 0.32 and 0.36; it helps Qwen3-1.7B-Base (+0.046), OLMo-2-7B (+0.045) and Qwen3-0.6B-Base (+0.029) less, with CIs excluding 0; it gives similar but non-significant gains for Qwen3-1.7B (+0.027) and SmolLM2-1.7B (+0.024); and it does nothing for OLMo-2-1B (+0.005). Sorted-format gains are therefore not specific to 8B. A4 is unchanged (+0.000), as expected of a count statistic, which ignores fact order.

Table: Sorted minus raw accuracy on L cells (exploratory; paired, same trials), with 95% CIs. † nf4. {#tab:app-cal-sorted}

| model | k=16 | k=22 | k=28 |
|---|---|---|---|
| Qwen3-0.6B-Base | +0.021 [−0.005, +0.046] | +0.006 [−0.018, +0.029] | +0.029 [+0.001, +0.059] |
| Qwen3-1.7B | −0.004 [−0.029, +0.022] | +0.029 [+0.005, +0.054] | +0.027 [−0.009, +0.061] |
| SmolLM2-1.7B | +0.010 [−0.014, +0.033] | +0.017 [−0.007, +0.041] | +0.024 [−0.010, +0.061] |
| Qwen3-1.7B-Base | +0.015 [−0.012, +0.043] | +0.016 [−0.010, +0.041] | +0.046 [+0.010, +0.080] |
| Qwen3-4B-Base † | +0.022 [−0.007, +0.050] | +0.029 [+0.002, +0.055] | +0.108 [+0.066, +0.149] |
| Qwen3-8B-Base † | +0.011 [−0.014, +0.038] | +0.030 [+0.003, +0.058] | +0.101 [+0.061, +0.144] |
| OLMo-2-1B | −0.016 [−0.042, +0.008] | +0.002 [−0.019, +0.024] | +0.005 [−0.018, +0.029] |
| OLMo-2-7B † | −0.007 [−0.032, +0.017] | +0.035 [+0.014, +0.057] | +0.045 [+0.014, +0.075] |
| A4 (trained, 4 layers) | −0.005 [−0.019, +0.007] | +0.002 [−0.003, +0.007] | +0.000 [+0.000, +0.000] |

The instructed variant adds a one-paragraph task header, at $k \in$ {10, 22}. @tab:app-cal-instr lists the instructed-minus-raw differences for every cell with at least 30 queries; this guard, added to the $k_L$ rule after the first sweep (deviation 4), is applied here to every class. Over the eight language models, the 32 L and G contrasts (8 models × $k \in$ {10, 22} × {L, G}, pooled over $t$) range from −0.083 to +0.042 (mean −0.0063, about −0.006), and 2 of their CIs exclude 0, both negative. Over all three classes, the 48 contrasts range from −0.083 to +0.053 (mean −0.0016), and 3 CIs exclude 0.

Table: Instructed minus raw accuracy by forcing class (exploratory; paired, pooled over $t$), with 95% CIs. Queries per cell are the same for every model: at $k = 10$, L 123, O 1,194 and G 1,217; at $k = 22$, L 1,258, O 94 and G 48. † nf4. {#tab:app-cal-instr}

| model | k | L | O | G |
|---|---|---|---|---|
| Qwen3-0.6B-Base | 10 | −0.008 [−0.038, +0.019] | −0.001 [−0.015, +0.015] | −0.005 [−0.022, +0.012] |
| Qwen3-0.6B-Base | 22 | −0.006 [−0.019, +0.009] | +0.021 [−0.041, +0.093] | +0.000 [−0.067, +0.059] |
| Qwen3-1.7B | 10 | +0.000 [−0.060, +0.065] | +0.005 [−0.018, +0.030] | −0.030 [−0.048, −0.011] |
| Qwen3-1.7B | 22 | −0.017 [−0.033, −0.002] | +0.021 [−0.068, +0.128] | −0.042 [−0.136, +0.000] |
| SmolLM2-1.7B | 10 | +0.008 [−0.040, +0.056] | −0.018 [−0.040, +0.003] | +0.016 [−0.001, +0.031] |
| SmolLM2-1.7B | 22 | +0.006 [−0.010, +0.023] | −0.011 [−0.095, +0.069] | −0.042 [−0.100, +0.000] |
| Qwen3-1.7B-Base | 10 | +0.041 [−0.007, +0.087] | −0.005 [−0.026, +0.016] | −0.003 [−0.021, +0.015] |
| Qwen3-1.7B-Base | 22 | +0.011 [−0.006, +0.029] | +0.053 [−0.016, +0.131] | +0.042 [+0.000, +0.107] |
| Qwen3-4B-Base † | 10 | +0.033 [−0.032, +0.097] | +0.023 [+0.004, +0.043] | −0.010 [−0.027, +0.009] |
| Qwen3-4B-Base † | 22 | −0.005 [−0.022, +0.013] | +0.032 [−0.034, +0.087] | −0.042 [−0.100, +0.000] |
| Qwen3-8B-Base † | 10 | −0.008 [−0.050, +0.034] | +0.002 [−0.018, +0.021] | +0.001 [−0.018, +0.020] |
| Qwen3-8B-Base † | 22 | −0.002 [−0.020, +0.018] | −0.032 [−0.109, +0.035] | +0.000 [+0.000, +0.000] |
| OLMo-2-1B | 10 | −0.033 [−0.079, +0.008] | +0.011 [−0.005, +0.027] | −0.006 [−0.022, +0.011] |
| OLMo-2-1B | 22 | +0.001 [−0.017, +0.018] | +0.000 [−0.045, +0.039] | −0.083 [−0.231, +0.036] |
| OLMo-2-7B † | 10 | −0.016 [−0.050, +0.015] | +0.000 [−0.021, +0.021] | −0.004 [−0.022, +0.014] |
| OLMo-2-7B † | 22 | +0.002 [−0.016, +0.020] | +0.021 [−0.025, +0.067] | +0.000 [−0.060, +0.065] |

#### Experiment-2 hypotheses
The preregistered H1, that L accuracy rises with $k$ for every model, is not supported. Its second part, that $k_L$ falls with scale, cannot be tested with the guard; without it, only two of the smallest models have a $k_L$, an artefact of 4 queries. H2, that any compounding appears only in the larger models, holds vacuously, since no model gains beyond the scramble control. H3, that held-out transfer separates A from N_iso in a model that compounds, cannot be tested.

## Trained networks {#app:trained}

#### Architecture and training
The networks of Experiment 3 are GPT-style decoders (pre-LN, rotary embeddings [@su2021rope], width 256, 8 heads, 1, 2, 4 or 6 layers; 3.17M parameters at 4 layers) with a 70-token vocabulary. Training ran for 14,000 AdamW steps [@loshchilov2019adamw] (β = 0.9/0.95, weight decay 0.1, clipping 1.0, seed 0) of $\lfloor 65536/L \rfloor$ sequences of length L, each step from a single (n, t), with 500 warmup steps and cosine decay to 10% of a $6 \times 10^{-4}$ peak ($10^{-3}$ for 1–2 layers); t ∈ 0–4 (t ≤ 2 at n=10, so that sequences fit 2,400 tokens). Each network thus saw about 0.91B training tokens (14,000 steps of up to 65,536 tokens). The target table is complete in training and the loss covers every answer token after the sequence's first fact, so the generator's cue size k, uniform in $[\lceil 0.15 n^{2} \rceil, \lfloor 0.45 n^{2} \rfloor]$, does not affect training. Four runs were interrupted by laptop standby or a scheduler change; three were resumed from their last checkpoint and one retrained from scratch, all reaching the preregistered 14,000 steps on a continuous learning-rate schedule (@app:deviations, rows 10–13).

#### Diets and preregistered criteria
Training sequences use orders {4, 5, 6, 7, 8, 10} and exclude S3, D4 and Q8, with fresh symbols per table and loss on answer tokens only. The mixed-environment diet (A1, A2, A4, A6, by layer count) draws each sequence from groups (0.5), identity-free Latin squares (0.25) or non-associative loops (0.25), so that history is informative about the environment. Three 4-layer controls use groups only (G4), groups from a fixed pool of 32 labelled tables per order (P4), and an exemplar diet of isomorphic-copy tasks and identity-free Latin squares with no group or identity (I4). Predicted classes were frozen before training. The method counts as validated if A4 is A or B-type, I4 is N_iso, and P4 and A1 are R or N; a run whose in-training remaining-fact accuracy at $n = 6$ never exceeds chance + 0.10 is *not learned* and excluded from validation. Each network was classified by the unchanged decision rule.

We applied the three preregistered Experiment-3 criteria (`exp3/PREREGISTRATION.md`) after the second audit. @tab:trained compares the predicted classes with the observed ones.

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

Under the preregistered learning-failure rule, every run except P4 exceeds chance + 0.10 (0.267) on in-training $n = 6$ remaining-fact accuracy, with maxima of 0.33–0.36. P4 never does (maximum 0.183): it memorised its pool (training loss 0.005) without transfer to fresh symbols, so it is *not learned*; we report it but exclude it from method validation. Because the count shortcut alone scores 0.33, this threshold cannot distinguish learning the task from learning the shortcut. P4's in-training value on the isomorphic-copy condition (1.000) is an artefact of pool mode, in which the generator returns memorised pool tables, and it is not used anywhere. Experiment 3's allowed adaptations (a 2×-steps retrain of A4, an M=8 rerun of P4) did not trigger: A4 passed the learning rule, and P4 did not generalise.

Method validation is *not met*: A4 and I4 are R, against predictions of A or B-type and of N_iso. The preregistration asks that each mismatch be attributed either to the method or to the training diet. The in-training evaluations, compared post hoc with the count shortcut below, attribute it to training, and on the identical trials the fingerprint returns the same class (R) for the networks as for the exact count learner C.

The depth prediction is not supported. Held-out G accuracy at $t = 4$ is 0.230 (A1), 0.226 (A2), 0.238 (A4) and 0.239 (A6). This is not monotone in depth, and the CIs overlap. comp is undefined for A1, A2 and A4, because their $\mathrm{acc}_L$ is below chance; for A6 it is negative, −0.22 [−3.24, +0.16].

#### The count-shortcut diagnostic (post hoc)
The diagnostic is post hoc. For each trial it defines the least-revealed set, the target words with the fewest occurrences as a result among the revealed cue facts, and it records three quantities: the within-trial constancy of the predictions, the share of predictions that fall in the least-revealed set, and the expected accuracy of learner C (uniform over that set) and of one-pass row/column elimination. At $n = 6$ the least-revealed set holds 0.28–0.34 of the target words, which is therefore the share expected from uniform guessing.

@tab:app-trained-count summarises all raw-format cells for each run, and @fig:trained shows the networks' G accuracy beside the exact learners. The learned networks other than A6 put 85–99% of their predictions in the least-revealed set and match learner C's accuracy to within 0.007; A4 matches it to within 0.001 on both the 1,800 main trials and the 500 cue-sweep trials. A6 is less constant (75% in the set) and beats C by +0.0135. P4, which is *not learned*, puts 28% of its predictions in the set, about the uniform rate, and falls 0.048 below C. The two pretrained models we diagnosed do not follow the shortcut: Qwen3-8B-Base and Qwen3-1.7B-Base put 31–42% of their predictions in the set, and Qwen3-8B-Base exceeds C on the main trials only because those include its near-perfect leak-control cell (0.970).

Table: Count-shortcut diagnostic over all raw-format cells (post hoc): share of predictions in the least-revealed set, and model minus learner C accuracy with its 95% CI. {#tab:app-trained-count}

| run | trials | share in least-revealed set | model − C [95% CI] |
|---|---|---|---|
| A1 (main) | 1,800 | 0.849 | −0.0065 [−0.0080, −0.0050] |
| A2 (main) | 1,800 | 0.932 | −0.0025 [−0.0037, −0.0014] |
| A4 (main) | 1,800 | 0.986 | −0.0008 [−0.0017, +0.0001] |
| A4 (cue sweep) | 500 | 0.990 | −0.0009 [−0.0017, −0.0001] |
| A6 (main) | 1,800 | 0.747 | +0.0135 [+0.0117, +0.0153] |
| G4 (main) | 1,800 | 0.980 | −0.0008 [−0.0018, +0.0001] |
| I4 (main) | 1,800 | 0.987 | −0.0007 [−0.0015, +0.0002] |
| P4 (main) | 1,800 | 0.283 | −0.0476 [−0.0518, −0.0432] |
| Qwen3-8B-Base † (main) | 1,800 | 0.409 | +0.0393 [+0.0310, +0.0482] |
| Qwen3-8B-Base † (cue sweep) | 500 | 0.421 | −0.0210 [−0.0302, −0.0120] |
| Qwen3-1.7B-Base (cue sweep) | 500 | 0.311 | −0.0588 [−0.0687, −0.0492] |

![G-cell accuracy (n=6) against the number of earlier tables t for the seven networks trained from scratch, beside the exact learners A, N_iso and C, with held-out types at t=4 shown at right. Every learned network lies close to the count-shortcut learner C, while P4, which was not learned, sits below chance.](fig/fig4_trained_transformers.png){#fig:trained}

@tab:app-trained-sweep follows A4 across the cue sweep. Its predictions become fully constant within a trial from $k = 16$ on, and its accuracy tracks learner C at every $k$ (within 0.004), rising from 0.196 to 0.334, while row/column elimination would rise from 0.238 to 0.921. On L cells alone, A4 goes from 0.17 at $k = 10$ through 0.21 and 0.26 to 0.33 at $k = 28$ (@tab:app-cal-k). Qwen3-8B-Base, for comparison, puts 37–48% of its predictions in the least-revealed set and falls below C at $k = 28$ (0.261 against 0.334).

Table: The A4 network across the cue sweep ($n = 6$, 100 trials per $k$, pooled over $t$; post hoc): within-trial constancy, share of predictions in the least-revealed set, and accuracy over all cells for A4, learner C, row/column elimination and Qwen3-8B-Base †. {#tab:app-trained-sweep}

| k | A4 constancy | A4 share in set | A4 acc | C acc | row/col acc | Qwen3-8B-Base acc |
|---|---|---|---|---|---|---|
| 6 | 0.92 | 0.95 | 0.196 | 0.200 | 0.238 | 0.194 |
| 10 | 0.97 | 0.99 | 0.214 | 0.213 | 0.313 | 0.208 |
| 16 | 1.00 | 1.00 | 0.245 | 0.245 | 0.486 | 0.238 |
| 22 | 1.00 | 1.00 | 0.272 | 0.274 | 0.715 | 0.260 |
| 28 | 1.00 | 1.00 | 0.334 | 0.334 | 0.921 | 0.261 |

The shortcut also accounts for the in-training plateau. @tab:app-trained-intrain applies learner C and row/column elimination to the in-training evaluation conditions, teacher-forced and generated with the training generator. Over all remaining facts, the shortcut scores about 0.33 at $n = 6$ and 0.25 at $n = 8$, while elimination would score about 0.70 at $n = 6$. The networks reach 0.29–0.36 at step 14,000. On the identical sequences, averaged over the five evaluation conditions of each order, A4, G4 and I4 sit within $\pm 0.006$ of the shortcut at $n = 6$ and $n = 8$ (single conditions, 64 sequences each, within about $\pm 0.02$), A1 and A2 sit below it, and A6 sits slightly above it (+0.015 at $n = 6$). Their first-fact accuracy after the cue (0.23 for A4 and 0.21 for I4 at step 14,000; chance 0.17) is near the shortcut's level (0.20) and well below elimination (0.30).

Table: Expected accuracy of learner C and of one-pass row/column elimination on the in-training evaluation conditions (post hoc; training generator, teacher-forced), over all remaining target facts and on the first fact after the cue. {#tab:app-trained-intrain}

| condition | n | t | C, all facts | row/col, all facts | C, first fact | row/col, first fact |
|---|---|---|---|---|---|---|
| in-distribution | 6 | 0 | 0.332 | 0.698 | 0.210 | 0.303 |
| held-out types | 6 | 0 | 0.331 | 0.703 | 0.203 | 0.307 |
| in-distribution | 6 | 4 | 0.334 | 0.705 | 0.192 | 0.290 |
| held-out types | 6 | 4 | 0.327 | 0.710 | 0.198 | 0.283 |
| scramble history | 6 | 4 | 0.331 | 0.710 | 0.181 | 0.306 |
| in-distribution | 8 | 0 | 0.247 | 0.625 | 0.120 | 0.222 |
| held-out types | 8 | 0 | 0.250 | 0.626 | 0.246 | 0.219 |
| in-distribution | 8 | 4 | 0.252 | 0.625 | 0.128 | 0.217 |
| held-out types | 8 | 4 | 0.250 | 0.624 | 0.252 | 0.223 |
| scramble history | 8 | 4 | 0.250 | 0.626 | 0.210 | 0.228 |

## Reasoning probe (exploratory) {#app:probe}

Experiment 4 (*exploratory*, not preregistered) asked whether direct answering limits Latin inference. Qwen3-1.7B answered one Latin-forced cell per trial under a chat prompt stating the Latin-square rule, so the probe measures inference given the rule. Decoding was greedy, with thinking on (at most 4,096 tokens) or off (at most 400); truncated thinking counts as incorrect. The planned design (four conditions × 30 trials, 1,536 thinking tokens) truncated all 13 thinking answers in its first condition, so we reduced it to 15 trials each of sorted and raw $k = 28$ (@app:deviations, row 17): the first 15 $t = 0$ trials of each format.

With thinking off, the model answered 4 of 15 sorted and 3 of 15 raw trials correctly. With thinking on, it answered 1 of 15 correctly in each format, and 14 of the 15 runs in each format were truncated. In all, 28 of the 30 thinking runs hit the 4,096-token budget. Both runs that finished (2,604 and 3,453 tokens) were correct, but paired by trial, no trial is correct only with thinking on. Six of the 30 non-thinking answers also reached their 400-token limit without an ANSWER line (sorted trials 6 and 8; raw trials 4, 6, 8 and 12). They were scored by the last candidate word mentioned, and all six were incorrect.

The probe is dominated by truncation. It neither shows nor excludes reasoning-enabled Latin inference at this scale, and it tests only whether a model can apply a *stated* rule, not induce one from histories.

## The compute–retrieve axis {#app:cr}

#### Background
A line of activation-space work reads the processing mode from activations. Demonstrations are compressed into a task vector [@hendel2023] or a function vector carried by a few attention heads [@todd2023fv], and the Jacobian lens identifies representations a model is poised to verbalise, which carry intermediate steps of silent reasoning while automatic processing proceeds without them, as in a global workspace [@gurnee2026workspace]. The compute–retrieve (CR) axis of our earlier preprint [@ramdan2026cr] probes mode directly: a mean-difference direction from 30 retrieval and 30 computation prompts, where "retrieve" means recall of facts stored in weights. That preprint notes as a limitation that its retrieval prompts are shorter and less complex than its computation prompts, leaving prompt complexity as a possible confound. Our stimuli remove this confound for in-context retrieval: in the $n = 6$ leak cell at $t = 1$ every query is retrievable, while in base at $t = 1$ the same format, token count and query position require inference. Behaviour differs sharply: accuracy over the queried cells is 0.53–0.97 in leak and 0.17–0.22 in base (0.17–0.21 on the subset of queries projected in Experiment 5).

#### Design of Experiment 5 (preregistered)
We ran the published detector code read-only, in bf16 rather than fp16: a mean-difference axis from the paper's 30 + 30 calibration prompts at the best of four layers, read at the last token of chat-template (primary) and raw inputs. The three bf16 models received all 26 queries per trial and the two nf4 models (†) 11–12 class-stratified queries, over 100 trials per condition. On Qwen3-0.6B-Base (bf16, uncached forwards) our forward pass reproduced the detector's hidden states exactly; the nf4 and CPU-embedding paths were not checked this way, and the cached path matched uncached forwards in an audit check (r ≥ 0.996; @app:impl). H1 predicts that base projects higher, toward "compute", than leak; it is supported if the trial-bootstrap CI excludes 0 in that direction in at least 3 of 5 models. H2 predicts a rise in the order L < O < G within base. The hypotheses were finalised minutes after a discarded 6-trial smoke test whose estimate pointed against H1; the direction was kept.

#### Results of Experiment 5
The detector separates its own calibration prompts well in every model we tested, with AUC 0.96–0.99 at the chosen layer. @tab:app-cr-h1 reports the preregistered H1 test for each model, together with the exploratory raw-format projection at the same layer. The detector layer is the best of four candidates by calibration AUC. In detector (chat-template) format, base − leak is significantly negative in three models and null in two, so H1 (base $>$ leak, with the CI excluding 0 in at least 3 of 5 models) is supported in 0 of 5. This outcome, a significant reversal, lies outside the preregistered interpretations, which covered only "supported" and "not supported" (@app:deviations, row 20). In raw format (exploratory), base $>$ leak with the CI excluding 0 in Qwen3-0.6B-Base, Qwen3-1.7B-Base and Qwen3-4B-Base, which would meet the criterion had raw format been primary, and base $<$ leak in Qwen3-1.7B (d = −0.28) and Qwen3-8B-Base (d = −1.68). H2 (projection rising from L to O to G within base) is null in four models and significantly reversed in Qwen3-1.7B-Base ($\rho$ = −0.113 [−0.176, −0.046]); the other four $\rho$ lie between −0.065 and −0.003. In an exploratory analysis, projection does not predict correctness within a condition (AUC 0.46–0.55, except 0.38 in Qwen3-1.7B-Base leak).

Table: Experiment 5: base minus leak projection on the CR axis at the detector layer, in detector format (preregistered H1) and raw format (exploratory), with trial-bootstrap 95% CIs, Cohen's d and the query-level AUC for base $>$ leak. Calibration AUC is at the chosen layer. † nf4. {#tab:app-cr-h1}

| model | layer (cal. AUC) | detector: base − leak [95% CI] | d | query AUC | raw: base − leak [95% CI] | raw d |
|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 26 (0.978) | −27.2 [−32.7, −21.5] | −1.34 | 0.19 | +5.8 [+3.8, +7.9] | +0.79 |
| Qwen3-1.7B-Base | 26 (0.964) | −158.5 [−173.0, −144.5] | −3.08 | 0.03 | +12.0 [+4.6, +19.6] | +0.44 |
| Qwen3-1.7B | 21 (0.993) | −0.06 [−1.5, +1.6] | −0.01 | 0.50 | −1.1 [−2.2, −0.1] | −0.28 |
| Qwen3-4B-Base † | 34 (0.973) | −11.1 [−15.5, −6.5] | −0.67 | 0.34 | +5.0 [+3.2, +6.7] | +0.77 |
| Qwen3-8B-Base † | 34 (0.991) | +0.70 [−6.7, +8.0] | +0.03 | 0.51 | −24.0 [−27.8, −20.1] | −1.68 |

#### Context vocabulary (Experiment 5b, exploratory)
Leak and base are matched in format, token count (281 context tokens in both) and query position, but not in vocabulary (@tab:app-cr-surface). In leak, the history table is written in the target's own six words, so every target line, including the queried pair and its answer, appears verbatim; in base, the history uses six other words. Leak contexts therefore have half the vocabulary, about 5× more occurrences of the query words, and a verbatim copy of every target line. In detector format the projected token is the template's newline after `assistant`, so the projection summarises the whole context.

Table: Surface statistics of the leak and base contexts. {#tab:app-cr-surface}

| statistic | leak | base |
|---|---|---|
| distinct content words | 6 | 12 |
| occurrences of the query words in the context | 45.5 | 9.5 |
| repeated 4-gram token fraction | 0.48 | 0.15 |
| query line present verbatim | always | never |

Experiment 5b (exploratory, not preregistered) followed the H1 result and a 20-trial audit pilot in which relabelling closed 53% and 71% of the gap. It was written down before its full run but after that pilot, so it is a full-sample estimate of an effect already seen, not an independent test. It changes only the history table. **leakRel** permutes the words of leak's history by a random derangement, which keeps its vocabulary but makes its answers inconsistent, so retrievability is the only difference from leak; **baseRel** writes base's history in the target's words, which keeps it non-retrievable. The share of the target's displayed lines found verbatim in the history is 1.00 for leak, 0.08 for leakRel, 0.11 for baseRel and 0.00 for base. $f_{\mathrm{vocab}} = (\mathrm{baseRel} - \mathrm{base})/(\mathrm{leak} - \mathrm{base})$ is the share of the base–leak gap reproduced by vocabulary alone, and leakRel − leak is the cleanest retrievability contrast (predicted $> 0$). The control used uncached forwards, all 26 queries and 100 trials on four models (8B was not run), so its base − leak values in @tab:app-cr-ctrl differ slightly from @tab:app-cr-h1.

Table: Experiment 5b vocabulary-overlap control at the detector layer (layer 26 for Qwen3-0.6B-Base and Qwen3-1.7B-Base, 21 for Qwen3-1.7B, 34 for Qwen3-4B-Base †), in detector and raw format, with 95% CIs. $f_{\mathrm{vocab}}$ is uninformative where the base–leak gap is small (Qwen3-1.7B) or where vocabulary moves the projection the other way (Qwen3-1.7B-Base, raw). {#tab:app-cr-ctrl}

| model | format | base − leak (d) | leak − baseRel [95% CI] | leakRel − leak [95% CI] ($d_z$) | $f_{\mathrm{vocab}}$ [95% CI] |
|---|---|---|---|---|---|
| Qwen3-0.6B-Base | detector | −27.26 (−1.34) | +8.95 [+3.51, +14.50] | −4.36 [−6.66, −2.05] (−0.36) | 0.67 [0.54, 0.85] |
| Qwen3-1.7B-Base | detector | −158.59 (−3.08) | +20.62 [+5.24, +35.83] | −11.83 [−20.03, −3.52] (−0.28) | 0.87 [0.79, 0.96] |
| Qwen3-1.7B | detector | −0.07 (−0.01) | −1.70 [−3.07, −0.35] | +1.34 [+0.59, +2.07] (+0.36) | 24.72 [−35.28, 35.26] |
| Qwen3-4B-Base † | detector | −10.50 (−0.64) | +3.78 [+0.08, +7.36] | −1.07 [−3.17, +0.95] (−0.10) | 0.64 [0.40, 0.99] |
| Qwen3-0.6B-Base | raw | +5.84 (+0.79) | −1.88 [−4.02, +0.18] | +1.21 [+0.84, +1.61] (+0.60) | 0.68 [0.48, 1.04] |
| Qwen3-1.7B-Base | raw | +12.04 (+0.44) | −19.93 [−27.49, −12.57] | +21.69 [+19.15, +24.19] (+1.67) | −0.66 [−2.02, −0.32] |
| Qwen3-1.7B | raw | −1.10 (−0.28) | −1.10 [−2.11, −0.03] | +1.75 [+1.26, +2.29] (+0.69) | 1.99 [0.85, 11.49] |
| Qwen3-4B-Base † | raw | +4.90 (+0.77) | −1.28 [−3.05, +0.59] | +0.98 [+0.42, +1.53] (+0.33) | 0.74 [0.53, 1.18] |

Vocabulary explains most of the reversal (@fig:cr). In the three reversed base models under the chat template, baseRel reproduces 64–87% of the base–leak gap ($f_{\mathrm{vocab}}$ 0.67 [0.54, 0.85] for Qwen3-0.6B-Base, 0.87 [0.79, 0.96] for Qwen3-1.7B-Base and 0.64 [0.40, 0.99] for Qwen3-4B-Base), and the remainder (leak − baseRel) is still reversed. With vocabulary held fixed (leakRel − leak, predicted > 0), the sign depends on model and wrapper. In detector format the contrast is reversed in Qwen3-0.6B-Base ($d_z$ −0.36) and Qwen3-1.7B-Base (−0.28), so the retrievable context projects *more* toward compute; it is null in Qwen3-4B-Base (−0.10) and in the predicted direction only in the post-trained Qwen3-1.7B (+0.36, and +0.69 in raw format). In raw format at the detector layer it is in the predicted direction in all four models ($d_z$ +0.33 to +1.67), although at other layers the sign varies.

The post-trained null is a cancellation. In Qwen3-1.7B, vocabulary overlap raises the projection (baseRel − base = +1.8) about as much as retrievability lowers it (leak − leakRel = −1.3), which is why the preregistered contrast is null there.

The axis therefore carries an in-context retrieval signal of the predicted sign in two places. In the post-trained model it is small ($d_z$ +0.36 in detector format), about the size of that model's vocabulary effect. In raw format it ranges from $d_z$ +0.33 to +1.67, and in Qwen3-1.7B-Base it exceeds the vocabulary effect. In the base models under the detector's chat template, by contrast, the vocabulary effect (baseRel − base) is 4–12× the retrievability contrast, and with vocabulary held fixed the retrievability contrast itself is reversed in Qwen3-0.6B-Base and Qwen3-1.7B-Base and null in Qwen3-4B-Base.

![Experiment 5b (exploratory): CR projection for each condition minus leak, in units of the pooled SD of the leak and base trial means, at the detector layer and in detector (chat-template) format, for four Qwen3 models (8B was not run). Writing a non-retrievable history in the target's words (baseRel) reproduces most of the base–leak gap, and making the retrievable history inconsistent (leakRel) lowers the projection in Qwen3-0.6B-Base and Qwen3-1.7B-Base, against the prediction, has no reliable effect in Qwen3-4B-Base, and raises it, as predicted, only in the post-trained Qwen3-1.7B.](fig/fig5_cr_axis_control.png){#fig:cr}

#### Interpretation
The CR axis separates its calibration prompts with AUC 0.96–0.99, but it does not track retrieval versus inference once format and length are matched: it is significantly reversed in three of five models, and its sign flips with prompt format. An exploratory control attributes 64–87% of the reversed gap to context vocabulary. With vocabulary matched, a retrieval signal of the predicted sign appears in the post-trained model (small, $d_z$ +0.36, comparable to its vocabulary effect) and without the chat template ($d_z$ +0.33 to +1.67). For representation-level claims about "modes" of processing, this is a cautionary case: a direction that separates prompt *types* need not track the *process*. A format- and length-matched behavioural contrast, like leak versus base, is a cheap test that any such probe should pass before it is interpreted, and vocabulary must be matched as well. Whether activation-space representations of processing mode would carry induced axioms across tables is untested.

## In-Context Algebra positive control {#app:ica}

We re-implemented the released model from its code without executing that code, and loaded the checkpoint with `weights_only=True`. A second, independent forward pass (explicit causal attention, real-valued RoPE, erf-GELU) agrees with the adapter on 3,200 of 3,200 answer positions, with a maximum absolute logit difference of $1.9 \times 10^{-4}$. @tab:app-ica-valid gives held-out final-fact accuracy on the model's own task. Under the upstream evaluation protocol on 1,024 fresh sequences, the released best checkpoint scores 0.959, below the 99.4% that its authors report [@todd2026ica]. Their figure is the maximum over up to about 290 training-time evaluations on one cached 512-sequence set, and `algebra_gpt_best.pt` was saved at that maximum; the 20k–100k checkpoints score 0.960–0.975 on our sequences, so every released checkpoint from 20k steps on, the best one included, scores 0.959–0.975. The shortfall is therefore consistent with snapshot selection on a fixed set, not with a port error. Full-sequence training loss is 0.917, against about 0.93 at the end of the upstream curve.

Table: Validation of the adapter on the model's own distribution: held-out final-fact accuracy. {#tab:app-ica-valid}

| setting | accuracy |
|---|---|
| shuffled vocabulary, 200 facts (512 sequences) | 0.973 |
| shuffled vocabulary, 100 / 50 / 20 facts | 0.91 / 0.83 / 0.66 |
| upstream protocol, 1,024 fresh sequences, best checkpoint | 0.959 |
| same sequences, checkpoints 100k / 40k / 20k / 10k | 0.960 / 0.975 / 0.971 / 0.835 |

The adapter maps each trial's words to distinct random symbols (four draws; no group identity maps to '0', as in training) and scores each query by forced choice over the target's $n$ symbols; only cells with at most 16 distinct words fit. Decision-rule steps 1–3 need only the base $t = 0$ cell; steps 4–5 need $t = 4$ and scramble histories, which do not fit the 16-token vocabulary. @tab:app-ica-steps gives these steps for every order, on accuracy and on mean $p_{\mathrm{gold}}$ (stimulus fact order, four symbol draws, bootstrap seeded as in the main analysis). Every row is non-R and non-B, with N, N_iso and A undetermined. Loop-forced cells score 0.90 ($n = 5$), 0.91 ($n = 6$) and 0.99 ($n = 8$), and O − G is +0.27 to +0.45 in every non-leak cell of the main design that fits the model (base $t = 0$ at $n = 5$, 6 and 8; base and held-out $t = 1$ at $n = 6$ and 8). The addendum-2d $k = 16$ cells keep O above G with a smaller gap (+0.20 to +0.22), with L/O/G accuracy 0.930/0.920/0.710 (base $t = 1$) and 0.909/0.917/0.713 (held-out $t = 1$), and the leak control scores 1.000 on every forcing class.

Table: In-Context Algebra model, decision-rule steps 1–3 at base $t = 0$ (preregistered at $n = 6$, primary, and $n = 8$, secondary; the $n = 5$ and addendum-2d rows apply the same steps to the other base $t = 0$ cells that fit the vocabulary), with trial-bootstrap 95% CIs and the number of L queries. Every row is classified non-R, non-B. {#tab:app-ica-steps}

| order and cue | measure | $\mathrm{acc}_L$ [95% CI] (L queries) | R threshold | G t0 [95% CI] | g(G t0) [95% CI] |
|---|---|---|---|---|---|
| n=6, k=10 | accuracy | 0.886 [0.820, 0.944] (114) | 0.417 | 0.637 [0.601, 0.675] | 0.65 [0.59, 0.74] |
| n=6, k=10 | $p_{\mathrm{gold}}$ | 0.852 [0.800, 0.897] (114) | 0.417 | 0.541 [0.519, 0.562] | 0.55 [0.50, 0.60] |
| n=8, k=16 | accuracy | 0.971 [0.909, 1.000] (34) | 0.375 | 0.663 [0.631, 0.699] | 0.64 [0.58, 0.71] |
| n=8, k=16 | $p_{\mathrm{gold}}$ | 0.864 [0.804, 0.922] (34) | 0.375 | 0.495 [0.479, 0.513] | 0.50 [0.46, 0.55] |
| n=5, k=6 | accuracy | 0.982 [0.952, 1.000] (111) | 0.450 | 0.531 [0.485, 0.574] | 0.42 [0.36, 0.48] |
| n=5, k=6 | $p_{\mathrm{gold}}$ | 0.901 [0.874, 0.928] (111) | 0.450 | 0.519 [0.496, 0.544] | 0.46 [0.42, 0.50] |
| n=6, k=16 (2d stimuli) | accuracy | 0.920 [0.900, 0.938] (1,106) | 0.417 | 0.704 [0.651, 0.758] | 0.71 [0.64, 0.79] |
| n=6, k=16 (2d stimuli) | $p_{\mathrm{gold}}$ | 0.903 [0.889, 0.917] (1,106) | 0.417 | 0.642 [0.604, 0.680] | 0.64 [0.59, 0.70] |

@tab:app-ica-pred lists the six preregistered predictions. Four are met; two, B-type and lower G accuracy on unseen types, are *not met* and are reported as such.

Table: Preregistered predictions for the In-Context Algebra model and their outcomes. {#tab:app-ica-pred}

| prediction | outcome | evidence |
|---|---|---|
| P1: not R at n=6 | met | $\mathrm{acc}_L$ 0.886 [0.820, 0.944] |
| P2: B-type at n=6 | not met | g(G t0) 0.65, below 0.85: non-R, non-B |
| P3: leak recall, G at least 0.90 | met | leak G 1.000 |
| P4: G above learner C (paired) | met | +0.411 [+0.369, +0.453] |
| P5: no gain from a differently labelled history (n=6 G, t1 − t0) | met | −0.025 [−0.071, +0.024] |
| P6: unseen order-8 types below seen types (n8 base t0) | not met | +0.025 [−0.035, +0.085]; pooled n=8 cells −0.005 [−0.036, +0.026] |

@tab:app-ica-n8 breaks $n = 8$ G accuracy down by target type. Z8 (= C8) and D4 were in the model's training mix; Z4×Z2, Z2×Z2×Z2 and Q8 were not. G accuracy depends on the specific group more than on whether it was seen: at base $t = 0$, Z2×Z2×Z2, whose smaller relative Z2×Z2 (= D2) was in training, scores 0.973, and Q8 scores 0.507. Latin- and loop-forced accuracy is 0.95–1.00 for seen and unseen types alike.

Table: G-cell accuracy at $n = 8$ by target type (seen in training: Z8, D4), with pooled seen / unseen accuracy. {#tab:app-ica-n8}

| cell | Z8 | D4 | Z4×Z2 | Z2×Z2×Z2 | Q8 | seen / unseen |
|---|---|---|---|---|---|---|
| base t0 | 0.627 | 0.671 | 0.673 | 0.973 | 0.507 | 0.648 / 0.673 |
| base t1 | 0.544 | 0.601 | 0.619 | 0.773 | 0.458 | 0.562 / 0.601 |
| held-out t1 | — | 0.575 | — | — | 0.464 | 0.575 / 0.464 |

At $n = 6$, base $t = 0$, the model scores 0.886 (L), 0.910 (O) and 0.637 (G), against 0.148, 0.212 and 0.226 for learner C and 0.563, 0.314 and 0.289 for one-pass row/column elimination; it beats both baselines by a wide margin in every class. @tab:app-ica-sweep places its cue sweep next to Qwen3-8B-Base, learner C and elimination on the same $t = 0$ trials. Row-sorting the facts makes no difference for this model (0.881 against 0.878 at $k = 16$). Interleaving all context facts in one random order (exploratory) also changes nothing: at $n = 6$, base $t = 0$, L, O and G score 0.886, 0.908 and 0.647, and the class is the same at $n = 5$, 6 and 8.

Table: Cue sweep at $n = 6$, $t = 0$: accuracy over all queried cells for the In-Context Algebra model and Qwen3-8B-Base † (raw and row-sorted facts), and the expected accuracy of learner C and one-pass row/column elimination on the same trials. {#tab:app-ica-sweep}

| k | model, raw | model, sorted | learner C | row/col elimination | Qwen3-8B-Base, raw | Qwen3-8B-Base, sorted |
|---|---|---|---|---|---|---|
| 6 | 0.689 | — | 0.200 | 0.239 | 0.199 | — |
| 10 | 0.744 | — | 0.216 | 0.311 | 0.207 | — |
| 16 | 0.878 | 0.881 | 0.244 | 0.491 | 0.253 | 0.264 |
| 22 | 0.964 | 0.964 | 0.271 | 0.729 | 0.277 | 0.289 |
| 28 | 0.998 | 0.998 | 0.325 | 0.931 | 0.292 | 0.407 |

@tab:app-ica-traj follows six of the 21 released numbered checkpoints (exploratory; $n = 6$, base $t = 0$, two symbol draws). The checkpoints at 0 and 5k steps are R, and those from 10k on are non-R, non-B; the upstream training loss drops sharply between those steps. After 20k, Latin and loop accuracy plateau at 0.86–0.91 while g(G t0) keeps rising, a graded change that a continuous index captures and the categorical classes do not.

Table: Training trajectory of the In-Context Algebra model across released checkpoints (exploratory), with the class from decision-rule steps 1–3 and cue-sweep accuracy at $k = 28$. {#tab:app-ica-traj}

| step | $\mathrm{acc}_L$ | O t0 | G t0 | g(G t0) | class | sweep k=28 |
|---|---|---|---|---|---|---|
| 0 | 0.167 | 0.155 | 0.162 | — | R | 0.172 |
| 5k | 0.351 | 0.392 | 0.250 | 0.45 | R | 0.575 |
| 10k | 0.807 | 0.840 | 0.483 | 0.49 | non-R, non-B | 0.930 |
| 20k | 0.860 | 0.885 | 0.550 | 0.55 | non-R, non-B | 0.988 |
| 40k | 0.868 | 0.896 | 0.600 | 0.62 | non-R, non-B | 0.988 |
| 100k | 0.886 | 0.911 | 0.635 | 0.65 | non-R, non-B | 0.998 |

The mechanism split is post hoc, added after the primary results. It pools the base $t = 0$ and $t = 1$ and held-out $t = 1$ cells of each order, and it splits G cells by whether the target is abelian and whether the transposed cell $b \cdot a$ is revealed in the cue. The baselines are expected accuracies on the same cells: learner N (uniform over Latin-feasible values), an exact loop-level learner (uniform over the values that a loop completion of the cue allows, computed with the stimulus generator's SAT engine) and learner C. G cells are by definition not loop-forced, so beating the loop-level learner requires group structure beyond identity and cancellation (@tab:app-ica-mech).

Table: Mechanism split of G cells (post hoc), pooled over the base $t = 0$/$t = 1$ and held-out $t = 1$ cells of each order: model accuracy (number of cells), share of predictions equal to the revealed transpose, and the expected accuracy of the loop-level learner and of learners N and C. {#tab:app-ica-mech}

| order | target | transpose | G acc (cells) | pred. = transpose | loop level | N / C |
|---|---|---|---|---|---|---|
| n=5 | abelian | hidden | 0.460 (463) | 0.000 | 0.456 | 0.322 / 0.290 |
| n=5 | abelian | revealed, = gold | 1.000 (70) | 1.000 | 0.474 | 0.304 / 0.029 |
| n=6 | abelian | hidden | 0.628 (872) | 0.000 | 0.430 | 0.300 / 0.272 |
| n=6 | abelian | revealed, = gold | 0.963 (214) | 0.963 | 0.415 | 0.281 / 0.063 |
| n=6 | non-abelian | hidden | 0.542 (1,947) | 0.000 | 0.417 | 0.301 / 0.239 |
| n=6 | non-abelian | revealed, $\neq$ gold | 0.583 (504) | 0.377 | 0.427 | 0.285 / 0.248 |
| n=6 | non-abelian | revealed, = gold | 1.000 (60) | 1.000 | 0.417 | 0.295 / 0.057 |
| n=8 | abelian | hidden | 0.610 (3,206) | 0.000 | 0.292 | 0.221 / 0.194 |
| n=8 | abelian | revealed, = gold | 0.925 (859) | 0.925 | 0.286 | 0.212 / 0.032 |
| n=8 | non-abelian | hidden | 0.501 (4,918) | 0.000 | 0.289 | 0.220 / 0.171 |
| n=8 | non-abelian | revealed, $\neq$ gold | 0.425 (760) | 0.458 | 0.290 | 0.216 / 0.168 |
| n=8 | non-abelian | revealed, = gold | 0.943 (601) | 0.943 | 0.283 | 0.215 / 0.038 |

Two mechanisms are visible. When a non-abelian target's transpose is revealed but differs from the queried cell, the model copies it 38% ($n = 6$) or 46% ($n = 8$) of the time, a commutativity shortcut that fails on non-abelian groups. With the transpose hidden, the model beats the loop-level learner at $n = 6$ (+0.148 [+0.130, +0.166]; 282 trials, 2,819 cells) and at $n = 8$ (+0.254 [+0.236, +0.273]; 300 trials, 8,124 cells), but not at $n = 5$ (+0.004 [−0.043, +0.048]; 53 trials, 463 cells). At $n = 6$ and $n = 8$, then, part of its associativity-forced accuracy goes beyond identity and cancellation. Pooled by target type, G accuracy is 0.561 for S3 and 0.694 for Z6 at $n = 6$, and ranges from 0.475 (Q8) to 0.868 (Z2×Z2×Z2) at $n = 8$.

## Implementation and provenance details {#app:impl}

This appendix collects implementation checks, provenance details and the audit history that the main text only summarises.

#### Forcing classes
Latin classes were computed by enumeration for n ≤ 5 and with a SAT solver [@ignatiev2018pysat] otherwise, loop classes by enumeration for n ≤ 6 and SAT otherwise, and group classes by enumerating all labelled tables of every group of order n. The exact learners' code recomputes the classes; it agreed with the stimulus labels on all 57,200 queried cells of the main design (0 mismatches), and SAT agreed with exact counts on 913 trials.

#### Scoring paths
Queries were batched on copies of each trial's prefix cache rather than scored singly as preregistered (@app:deviations, row 22). Each trial's prefix is encoded once. Queries with equal suffix length are scored in chunks of up to B on a copy of the prefix KV cache; from 2026-10-07 21:25 a one-query chunk is instead scored on the prefix cache itself, which is then cropped back (identical arithmetic).

B is set per trial from a KV-memory budget, and from 2026-10-07 13:16 it is halved after an out-of-memory error; it therefore varies with context length and hence with condition. For example, OLMo-2-7B used B = 8 at $t = 0$ and B = 1 at $t = 4$. From 2026-10-07 13:16 each scoring process was capped at 60% of GPU memory (62% for OLMo-2-7B).

On the Experiment-1 models, batched and sequential scoring agreed on 94–98% of argmax decisions (max $|\Delta p_{\mathrm{gold}}|$ 0.05, near-ties). Some result files mix revisions of the scoring script: `run.py` was revised at least three times, and the Qwen3-8B-Base main file, the last 216 trials of the Qwen3-4B-Base main file and the first 50 OLMo-2-7B calibration trials were scored with versions that were not archived (@app:deviations, row 8). The four $n = 6$ cells that enter Qwen3-1.7B-Base's primary contrast were re-scored with the final code and B = 1. Per-cell G accuracy moved by at most 0.005, and t4 − scramble was −0.006 [−0.034, +0.021] against −0.007 [−0.037, +0.020] originally (`exp2/offchk_result.md`).

#### CPU-offload path for the 7–8B models
For Qwen3-8B-Base and OLMo-2-7B, token embeddings were looked up on the CPU from the bf16 checkpoint weights, and the output projection was computed on the CPU in float32. We compared this path with standard scoring on two smaller models (@tab:app-offload); it was not compared on OLMo-2-7B or Qwen3-8B-Base themselves.

Table: CPU-offload scoring path against standard scoring (batch 1; `exp2/offchk_result.md`). {#tab:app-offload}

| model | trials | argmax agreement | mean abs. $\Delta p_{\mathrm{gold}}$ (max) | accuracy, offload vs standard |
|---|---|---|---|---|
| Qwen3-1.7B-Base | 72 | 2,102 of 2,288 (91.9%) | 0.007 (0.05) | 0.199 vs 0.201 |
| Qwen3-4B-Base, nf4 | 36 | 1,083 of 1,144 (94.7%) | 0.004 (0.02) | 0.199 vs 0.197 |

The 4B check is the configuration used for 8B. Taken together, batched, offloaded and version-mixed scoring were each checked against sequential scoring: argmax agreement was 92–98%, with near-ties in bf16; the Qwen3-1.7B-Base primary contrast, re-scored with one code version, gave −0.006 against −0.007; and classes are identical on accuracy and on $p_{\mathrm{gold}}$.

#### Quantisation control
The preregistered control (Qwen3-1.7B-Base, nf4 against bf16, all 1,800 trials) *failed* its tolerance (@app:deviations, row 6). nf4 lowered G accuracy in all 17 cells that contain G queries, by 0.020–0.067 (pooled −0.034; mean of per-cell differences −0.040), and per-query argmax agreement was 56%, so nf4 changed 44% of per-query answers. The class (R) is unchanged, and the compounding contrast is null in both precisions: at $n = 6$, t4 − scramble is −0.007 [−0.037, +0.020] in bf16 and +0.006 [−0.022, +0.035] in nf4. nf4 therefore did not create a spurious effect, and it preserved strong context use (leak 0.63 → 0.59). What this control cannot show is whether nf4 would hide a weak real effect at 4B, 7B or 8B, because at 1.7B there was nothing to hide. The low bias (pooled −0.034) applies to G accuracy; pooled L accuracy moved by −0.024 and O accuracy by +0.032.

#### CR-axis projections
The cached forward path used for every Experiment-5 projection was checked in an audit against full uncached forwards. The two agreed with r ≥ 0.996 and a median relative difference ≤ 3.5%, and sample condition differences were reproduced within 1.3 units.

In Experiment 5b, Qwen3-0.6B-Base ran one query per forward pass. The other three models ran the 26 queries of a trial-condition in one batched forward pass. Every row has the same token length, so no padding occurs, and a three-trial check reproduced the per-query projections exactly.

#### In-Context Algebra model
- **Checkpoint:** `algebra_gpt_best.pt` from the authors' project page, sha256 9c0ee347…a265d9.
- **Code:** read at commit 532a6a2, never executed. The model was re-implemented and loaded with `weights_only=True`.
- **Independent check:** a second forward pass (explicit causal attention, real-valued RoPE, erf-GELU) agreed with the adapter on 3,200 of 3,200 answer positions (max $|$logit difference$|$ $1.9 \times 10^{-4}$).
- **Interpreter:** Windows Smart App Control blocked the GPU environment's virtual-environment launcher on 2026-10-11, so these runs used the base CPython 3.12 interpreter with the same installed packages (torch 2.7.1+cu128).

#### Audits and corrections
Two adversarial audits (2026-10-07, 2026-10-09) re-derived every Experiment 1–5 number that existed at the time; later verification passes (2026-10-09 to 2026-10-11) re-derived the results produced after them (learner C, the count-shortcut diagnostic, the offload and re-scoring checks, the full Experiment 5b run and the addendum-2d run), checked the revised text, and checked Experiment 6 (2026-10-11). These are automated self-checks by the AI system that ran the analyses, not independent human review (see Statements). The audits recomputed the forcing classes of all 2,800 Experiment 1–2 stimulus trials with independent SAT and enumeration code (0 mismatches), as did the same checker on the 12,000 queries of the 600 later addendum-2d trials. They corrected an inflated copy-bias statistic, an unguarded partial-file value, order-dependent bootstrap seeds, stale ranges and model counts, an unapplied learning-failure rule, an overstated "dense-context Latin inference" reading of the trained networks, and the post-hoc guard that had removed the preregistered $k = 16$ follow-up. None changed a fingerprint classification.

## Deviations from preregistration {#app:deviations}

@tab:app-dev-1, @tab:app-dev-2, @tab:app-dev-3 and @tab:app-dev-4 list every deviation from the preregistrations. Dates are in 2026 and given as month-day, and times are local. Apart from the $\geq 30$-L-query guard added to the $k_L$ rule (row 4), none of these changes altered a stimulus, threshold or decision rule.

Table: Deviations from preregistration, rows 1–5. {#tab:app-dev-1}

| row | prereg | planned | done | when |
|---|---|---|---|---|
| 1 | Exp. 1 | 7–8B models | Qwen3-0.6B/1.7B and SmolLM2-1.7B; extended to 8B in Exp. 2 | 10-06 |
| 2 | Exp. 1 | 200 trials per cell (c10) | 100 | 10-06 |
| 3 | Exp. 1 | n=8 optional | promoted to secondary; abc, lesion and fine-tuning cells deferred | 10-06 |
| 4 | Exp. 2 (2a) | $k_L$ rule | $\geq 30$-L-query guard added after the Qwen3-0.6B-Base sweep; contrary to the 10:16 log entry, it changes the status of that model and of Qwen3-1.7B-Base (k=6: 2/4 = 0.50, so $k_L$ = 6 and 2d applies); 2d run on 10-09, both remain R ($\mathrm{acc}_L$ 0.18, 0.22) | 10-07 10:16 |
| 5 | Exp. 2 | no sorted variant | `kso_` stimuli generated at 10:32:57, 22 s after Qwen3-1.7B-Base calibration scoring began and before any of its output was read (logged 10:36); the sorted rows of Qwen3-0.6B-Base, Qwen3-1.7B, SmolLM2-1.7B and Qwen3-1.7B-Base, calibrated before the files existed, scored on 10-09 with the final code | 10-07 / 10-09 |

Table: Deviations from preregistration, rows 6–12. {#tab:app-dev-2}

| row | prereg | planned | done | when |
|---|---|---|---|---|
| 6 | Exp. 2 (2c) | tolerance $\leq 0.03$ | FAILED (max per-cell G difference 0.067; argmax agreement 56%) | 10-08, about 17:16 |
| 7 | Exp. 2 (2b) | 8B and 7B in nf4 | also CPU embeddings and an fp32 CPU head | 10-07 / 10-08 |
| 8 | Exp. 2 | a single scoring procedure | `run.py` revised at least three times; some result files mix versions; the 8B file, the last 216 trials of the 4B main file and the first 50 OLMo-2-7B calibration trials scored with unarchived versions | 10-07 13:16, 17:43–21:21, 21:25 |
| 9 | Exp. 2 (2d) | conditional k=16 | not run at the time (excluded by the guard of row 4); run on 10-09 for Qwen3-0.6B-Base and Qwen3-1.7B-Base with new-seed stimuli | 10-09 |
| 10 | Exp. 3 | 14k steps, one run | A4 interrupted at about 3,200 steps and resumed from step 2,000 | 10-07 13:26–14:29 |
| 11 | Exp. 3 | as above | A2 stopped by its cap at step 1,759 after standby; retrained from scratch with the same seed | 10-07 22:52 to 10-08 21:22 |
| 12 | Exp. 3 | as above | G4 stopped by its cap at step 9,934 during standby; scored, then resumed to 14,000 | 10-08 to 10-09 16:57 |

Table: Deviations from preregistration, rows 13–20. {#tab:app-dev-3}

| row | prereg | planned | done | when |
|---|---|---|---|---|
| 13 | Exp. 3 | as above | A6 stopped by its cap at step 13,471; scored, then resumed to 14,000 | 10-09 11:49–17:04 |
| 14 | Exp. 3 | `--minutes` cap 110/150 (README) | 150, 60, 90 or 240, later 90, finally none; from 10-09 the cap counts active time only | various |
| 15 | Exp. 3 | learning-failure rule; decision criteria | applied in @app:trained (after the second audit) | 10-09 |
| 16 | Exp. 3 | C4 optional | not run | allowed |
| 17 | Exp. 4 design note (exploratory, not preregistered) | 4 cells × 30 trials, 1,536 tokens | 2 cells × 15 trials, 4,096 tokens (chosen after seeing the v1 truncations) | 10-08 20:52 |
| 18 | Exp. 5 | 12 queries per trial for 4B and 8B | 11 in 20 of 400 trials | 10-09 |
| 19 | Exp. 5 | forward-path equivalence | checked only for the uncached forward; the cached path was validated later by the audit | 10-09 |
| 20 | Exp. 5 | interpretations for "supported" and "not supported" | the observed outcome (significant reversal in 3 of 5 models) was not covered; reported as such | 10-09 |

Table: Deviations from preregistration, rows 21–23. {#tab:app-dev-4}

| row | prereg | planned | done | when |
|---|---|---|---|---|
| 21 | — | — | post-hoc additions: learner C, the count-shortcut diagnostic, the answer-bias diagnostic, Experiment 5b; the type-matched held-out check and the multiple-comparisons check (@app:typematched; @app:mult) | 10-07 / 10-09 / 10-12 |
| 22 | Exp. 1 (Exp. 2 by reference) | each query scored on the prefix cache, cropped back after each query | queries of equal suffix length batched (up to B per chunk) on copied prefix caches; 2–6% of near-tie argmax decisions differ from sequential scoring | 10-06 |
| 23 | Exp. 6 | adapter validation exactly as the upstream protocol, expected at least 0.99; steps 1–3 bootstrap seeded as in the main analysis | validation used the same generator and final-token argmax but 1,024 fresh sequences: 0.959, explained by checkpoint selection; the first analysis run seeded steps 1–3 differently and was corrected (CI endpoints moved by at most 0.01); mechanism split and loop-level baseline added after the primary results; the B-type and unseen-type predictions were not met and are reported as such | 10-11 |

Two further notes apply. The resume path overwrites `config.json` with the arguments of the resume invocation; we checked these by hand against the original runs, and they match on every setting that affects the schedule and data. The learning-rate schedule is continuous across every resume, with every logged learning rate matching the planned schedule and no loss discontinuity, so the continuations in rows 10, 12 and 13 (row 11 is a from-scratch retrain with the same seed), although not among the preregistered allowed adaptations, only complete the preregistered schedule.
