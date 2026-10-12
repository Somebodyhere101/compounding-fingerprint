## Results {#sec:results}

Every run is complete. All numbers come from auto-generated result tables, the full table for every model, network and idealised learner except Asl is in @app:main, and post hoc and exploratory analyses are labelled where they appear.

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

On associativity-forced (G) cells the signatures differ sharply (@fig:ideal). The notebook learner N stays flat (0.29 at t=0, 0.28 at t=4, 0.29 on held-out types); exemplar matching (N_iso) rises gradually, through 0.30, 0.69, 0.85 and 0.98 at t = 0, 1, 2 and 4, but stays at 0.30 on held-out types (shape +0.35 [+0.26, +0.45], $w$ about 0); and abstraction (A) steps to 1.00 at t=1 and keeps that level on held-out types (shape 0.000, $w$ = 1.00). B and Bh are classified B-type at step 3, before the compounding contrast is consulted. The compounding contrast, t=4 minus scramble, is about +0.70 for both A and N_iso, so only held-out transfer separates them cleanly: held-out minus scramble is +0.70 for A against +0.03 for N_iso.

Held-out targets are non-abelian types only (S3 at $n = 6$; D4 or Q8 at $n = 8$) with abelian histories, while the base and scramble baselines mix target types. A post-hoc check restricts the baselines to trials with held-out target types: at $n = 6$, held-out minus type-matched scramble is +0.695 [+0.654, +0.735] for A and +0.033 [−0.005, +0.073] for N_iso, and with the same target types N_iso gains +0.681 [+0.628, +0.721] when the target's type is in the history (@app:typematched).

![Accuracy of the exact idealised learners on associativity-forced (G) cells against the number of earlier complete tables t, at n=6 (left) and n=8 (right), with held-out types at t=4 shown as diamonds. Abstraction (A) and exemplar matching (N_iso) both rise with t, but only A transfers to held-out types; the notebook learner (N) stays flat, and Bayes with the true prior (B) is at ceiling from t=0.](fig/fig1_ideal_learners.png){#fig:ideal}

### Pretrained models: class R, and no detectable gain beyond the scramble {#sec:r-llm}

All nine checkpoints (ten model–precision configurations) from three families are classified R at both n=6 and n=8, on both accuracy and $p_{\mathrm{gold}}$ (@tab:recall): accuracy on Latin-forced cells at t=0 ($\mathrm{acc}_L$) is 0.14–0.29, several values are at or below chance (0.167), and every value is below the preregistered threshold of 0.417.

#### No detectable gain beyond the scramble

At n=6, t4 − scramble lies between −0.019 and +0.026 (Qwen3-8B-Base: −0.008 [−0.038, +0.022]), and every 95% CI includes 0 (@fig:llm). The largest upper bound, +0.056 (Qwen3-4B-Base), is an order of magnitude below the gains of N_iso (+0.718) and A (+0.701). Held-out minus scramble lies between −0.04 and +0.02. Type-matched (post hoc), held-out minus scramble at $n = 6$ lies between −0.038 and +0.012 for the ten configurations, every CI including 0, and having the target's type in the history adds at most +0.033 (every CI including 0), against +0.681 for N_iso. The one raw held-out contrast that excluded 0, Qwen3-0.6B's −0.040 [−0.073, −0.007] against all scramble trials, is +0.001 [−0.037, +0.038] type-matched, so it reflected the difficulty of S3 targets rather than the history.

Two raw increases from t=0 to t=4 have CIs excluding 0, and both are no longer distinguishable from zero against the scramble history: OLMo-2-1B at n=6 gains +0.042 [+0.011, +0.073] from t=0 to t=4 but +0.020 [−0.008, +0.051] relative to scramble, and Qwen3-0.6B at n=8 gains +0.031 [+0.014, +0.049] but +0.008 relative to scramble, with a CI that includes 0. OLMo-2-1B's increase (+0.042) also does not survive a post-hoc Bonferroni correction across the 20 model-by-order contrasts (@app:mult); Qwen3-0.6B's at $n = 8$ (+0.031) does. Both raw gains are consistent with effects of context length alone.

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

#### Recall depends on family more than on size

When the history contains the target's own labelled table (the leak control), G-cell accuracy is 0.52–0.65 for the Qwen3 models up to 1.7B (0.59 for nf4 Qwen3-1.7B-Base), 0.95 for Qwen3-4B-Base, 0.96 for Qwen3-8B-Base, 0.44 for SmolLM2-1.7B, and 0.31 and 0.37 for OLMo-2-1B and OLMo-2-7B: the larger Qwen3 models retrieve almost perfectly and still stay below the Latin-inference threshold. A post-hoc diagnostic finds a bias towards answering G cells with one of the query's own words, which pulls G accuracy below chance in some models (@app:main).

![Compounding contrasts on G cells at n=6: t=4 minus the length-matched scramble history for seen types (left) and for held-out types (right), with 95% trial-bootstrap CIs, for nine pretrained checkpoints and three idealised learners. Every pretrained interval on seen types includes 0, whereas A and N_iso sit near +0.70 and only A keeps that gain on held-out types.](fig/fig2_llm_contrasts.png){#fig:llm}

### Calibration: Latin accuracy stays near chance up to 8B {#sec:r-cal}

On the cue sweep (n=6; k = 10 → 28 revealed cells of 36), Latin-forced accuracy pooled over t stays close to chance (0.167) for every pretrained model and never comes near the preregistered $k_L$ threshold of 0.5 at any k with at least 30 L queries (@tab:app-cal-k; @fig:cal); only Qwen3-8B-Base (+0.084 [+0.010, +0.153]) and OLMo-2-7B (+0.069 [+0.004, +0.128]) rise marginally from k=10 to k=28. With the guard, $k_L$ is undefined for all eight calibrated models, which are therefore *non-inferential* on this format (the post-trained Qwen3-0.6B was not swept). Without it, Qwen3-0.6B-Base and Qwen3-1.7B-Base qualify at k=6 (2 of 4 L queries correct), which triggers the conditional k=16 step of addendum 2d; run as preregistered after the final audit, it leaves both R ($\mathrm{acc}_L$ 0.177 [0.154, 0.201] and 0.219 [0.196, 0.241]; @app:calib).

**Missing information does not explain the failure (post hoc).** At k=28, 98% of the missing cells are Latin-forced (784 of 800), one-pass row/column elimination leaves a single candidate for 86% of those (675 of 784; `exp2/k28_elimination.md`), and choosing uniformly among the remaining candidates it would score 0.92 over all missing cells.

Two *exploratory* format variants, row-sorted facts and an instructed header, are reported in @app:calib and summarised in @sec:discussion. Of the preregistered Experiment-2 hypotheses, H1 is not supported, H2 holds only vacuously and H3 cannot be tested.

### Trained transformers learn a counting shortcut {#sec:r-trained}

The seven networks trained from scratch (@sec:trained; 3.2M parameters at 4 layers, about 0.91B training tokens each) are all classified R and all fail the recall (leak) control, with leak G accuracy of 0.08–0.24 (@tab:app-main-class): they neither recall nor infer at sparse cues. In a post-hoc diagnosis, the learned networks answer each trial with essentially one word, the target word least represented among the revealed facts (`exp3/heur/`): in A4, 98.6% of predictions fall in that least-revealed set, which across cells holds about 30% of the target words. A4's accuracy equals that of the count shortcut C to within 0.001 on all 1,800 main trials (−0.0008 [−0.0017, +0.0001]) and on the cue sweep, and A1, A2, G4 and I4 match C to within 0.007 (@fig:trained); only the 6-layer A6 beats it, by +0.014 [+0.012, +0.015], and P4, which is not learned, does not follow it (@app:trained).

The shortcut also explains the in-training plateau (0.33 at n=6 for the shortcut, 0.29–0.36 for the networks at step 14,000, 0.70 for cell-level row/column elimination) and A4's rise on L cells of the cue sweep (0.17 at k=10 to 0.33 at k=28); over all cells of the sweep, A4's accuracy matches the shortcut's to within 0.004 at every k (@app:trained). The networks therefore learned a global symbol-count statistic, a weak consequence of the Latin property, rather than Latin constraints.

Every run except P4, which memorised its pool without transfer to fresh symbols, exceeds the preregistered learning-failure threshold of chance + 0.10 (0.267) on in-training n=6 remaining-fact accuracy; but the count shortcut alone scores 0.33 (post hoc), so this threshold does not distinguish learning the task from learning the shortcut. Method validation is *not met*: A4 and I4 are R, against predictions of A or B-type and of N_iso (@tab:trained), a mismatch that the in-training evaluations, compared post hoc with the count shortcut, attribute to training rather than to the method; the fingerprint gives the networks the same class (R) as the exact count learner C. The preregistered depth prediction is not supported (@app:trained).

### Positive control: the In-Context Algebra model {#sec:r-ica}

Table: The released In-Context Algebra model under decision-rule steps 1–3 (base, t=0, accuracy). The R threshold is 1/n + 0.25, and g(G t0) ≥ 0.85 would make the model B-type. Every row is classified non-R, non-B, and the class is the same on mean $p_{\mathrm{gold}}$. {#tab:ica}

| cells | $\mathrm{acc}_L$ [95% CI] | R threshold | O | G [95% CI] | g(G t0) [95% CI] |
|---|---|---|---|---|---|
| n=6, k=10 | 0.886 [0.820, 0.944] | 0.417 | 0.910 | 0.637 [0.601, 0.675] | 0.65 [0.59, 0.74] |
| n=8, k=16 | 0.971 [0.909, 1.000] | 0.375 | 0.988 | 0.663 [0.631, 0.699] | 0.64 [0.58, 0.71] |
| n=5, k=6 | 0.982 [0.952, 1.000] | 0.450 | 0.895 | 0.531 [0.485, 0.574] | 0.42 [0.36, 0.48] |
| n=6, k=16 (2d) | 0.920 [0.900, 0.938] | 0.417 | 0.920 | 0.704 [0.651, 0.758] | 0.71 [0.64, 0.79] |

The model is neither R nor B-type (@tab:ica; @fig:ica). At n=6, $\mathrm{acc}_L$ is 0.886 [0.820, 0.944] against a threshold of 0.417 and loop-forced cells score 0.91, while associativity-only (G) cells score 0.637 [0.601, 0.675], so g(G t0) = 0.65 [0.59, 0.74] is below the B-type threshold of 0.85. The same holds at n=8 and n=5, on the addendum-2d k=16 stimuli and on $p_{\mathrm{gold}}$, and O − G is +0.27 to +0.45 in every non-leak cell of the main design that fits the model. This is the first real network for which the fingerprint returns a class above R, and its profile is consistent with its authors' mechanistic analysis: identity, cancellation and copying are near ceiling, and associativity is weakest.

Four of the six preregistered predictions are met (not R; leak recall 1.000; G accuracy above learner C, +0.411 [+0.369, +0.453]; no benefit from a differently labelled history). Two are *not met*: the model is not B-type, and G accuracy is not lower on the order-8 types held out from its training (+0.025 [−0.035, +0.085]); accuracy depends more on the specific group than on whether it was seen (Z2×Z2×Z2 and Q8, both unseen, score 0.97 and 0.51; @tab:app-ica-n8). On the t=0 cue sweep its accuracy over all cells reaches 0.998 at k=28, against 0.29 for Qwen3-8B-Base, 0.33 for learner C and 0.93 for one-pass row/column elimination (@fig:ica).

Interleaving all context facts leaves the class unchanged, and across released training checkpoints the class changes from R to non-R between 5k and 10k steps (exploratory; @app:ica). In a post-hoc mechanism split, the model partly answers by commutative copying: when a non-abelian target's transposed cell is revealed but differs from the queried cell, it returns the transpose in 38% (n=6) to 46% (n=8) of cases. With the transpose hidden, it beats the exact loop level (@sec:classes) at n=6 (+0.148 [+0.130, +0.166]) and n=8 (+0.254 [+0.236, +0.273]) but not at n=5 (+0.004 [−0.043, +0.048]).

![(a) Accuracy on L-, O- and G-forced cells at n=6, base t=0, for the released In-Context Algebra model, Qwen3-8B-Base, the trained network A4 and the idealised learners N, C and B. (b) Accuracy over all queried cells on the t=0 cue sweep for the In-Context Algebra model and Qwen3-8B-Base, against learner C and one-pass row/column elimination.](fig/fig6_in_context_algebra.png){#fig:ica}

### The compute–retrieve axis {#sec:r-cr}

Experiment 5 (*preregistered*) tested the compute–retrieve detector of our earlier preprint [@ramdan2026cr] in five Qwen3 models on the leak and base conditions at t=1, which share format, token count and query position but differ in retrievability (accuracy 0.53–0.97 in leak, 0.17–0.22 in base). Its H1, that base projects higher than leak (toward "compute"), is supported in 0 of 5 models against a criterion of at least 3, and three models are significantly reversed. This holds in the detector's chat-template format; in raw format (exploratory), base significantly exceeds leak in three models. The conditions are not vocabulary-matched: an exploratory control attributes 64–87% of the reversal to context vocabulary, and with vocabulary matched a retrieval signal of the predicted sign appears in the post-trained model and without the chat template (@app:cr).
