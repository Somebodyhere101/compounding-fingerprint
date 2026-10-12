## Experimental setup {#sec:setup}

All model inference and training ran on one shared laptop RTX 5070 GPU (8 GB); the exact learners and analyses ran on the CPU. Experiments are numbered as in their preregistrations: 1 and 2 (pretrained models and calibration), 3 (trained networks), 4 (an exploratory reasoning probe), 5 and 5b (the CR axis) and 6 (the positive control).

### Models and scoring {#sec:models}

We scored nine checkpoints (ten model–precision configurations) from three families (@tab:models): Qwen3 [@qwen3], base and post-trained; SmolLM2-1.7B-Instruct [@smollm2], a Llama-architecture model; and the fully open OLMo-2 [@olmo2].

Table: Pretrained models and the experiments that used them (2a: calibration sweep only). All checkpoints are ungated and Apache-2.0 licensed. † nf4, with the caveat of the failed quantisation control; Qwen3-1.7B-Base ran in nf4 on the main trials only. {#tab:models}

| model | family | training | precision | experiments |
|---|---|---|---|---|
| Qwen3-0.6B | Qwen3 | post-trained | bf16 | 1 |
| Qwen3-1.7B | Qwen3 | post-trained | bf16 | 1, 2a, 4, 5, 5b |
| Qwen3-0.6B-Base | Qwen3 | base | bf16 | 2, 5, 5b |
| Qwen3-1.7B-Base | Qwen3 | base | bf16; nf4 † | 2, 5, 5b |
| Qwen3-4B-Base | Qwen3 | base | nf4 † | 2, 5, 5b |
| Qwen3-8B-Base | Qwen3 | base | nf4 †, CPU embeddings and head | 2, 5 |
| SmolLM2-1.7B-Instruct | SmolLM2 | post-trained | bf16 | 1, 2a |
| OLMo-2-0425-1B | OLMo-2 | base | bf16 | 2 |
| OLMo-2-1124-7B | OLMo-2 | base | nf4 †, CPU embeddings and head | 2 |

Prompts are the raw context plus the query, without a chat template. Batched scoring on copied prefix caches, a CPU offload of the embeddings and float32 output head for the 7B and 8B models, and result files that mix revisions of the scoring script departed from the preregistered procedure. Each was checked against standard scoring on a subset only: batching on the Experiment-1 models and the offload on two smaller models (argmax agreement 92–98%), and the script revisions on one model's primary contrast (@app:impl; limits in @sec:discussion). nf4 quantisation [@dettmers2023qlora] was needed to fit the 4B, 7B and 8B models. Its *preregistered* control on Qwen3-1.7B-Base, which required an unchanged class and every per-cell G accuracy within 0.03 of bf16, **failed** (G accuracy 0.020–0.067 lower in all 17 cells with G queries), although the class and the null t4 − scramble contrast were unchanged (@app:impl); as preregistered, all nf4 results (†) carry the caveat that G accuracy is biased low by about 0.04 and per-query predictions are unstable.

### Calibration {#sec:calib}

Because compounding presupposes Latin inference, Experiment 2a swept cue density (n=6, base histories, t ∈ {0, 4}, k ∈ {6, 10, 16, 22, 28}, 50 new trials per (t, k)) on eight models. The Latin threshold $k_L$ is the smallest k at which L accuracy, pooled over t, reaches 0.5; a model without one is *non-inferential* on this format. A guard requiring at least 30 pooled L queries per k was added after the first sweep had been seen (@app:calib). Two *exploratory* format variants are not used for classification, and learner C and one-pass row/column elimination, which guesses uniformly among the values absent from the cell's row and column, serve as *post hoc* references. An exploratory reasoning probe (Experiment 4), in which Qwen3-1.7B answered Latin-forced cells with thinking on or off, is reported in @app:probe.

### Transformers trained from scratch {#sec:trained}

To test whether the fingerprint recovers the class implied by a known training diet (Experiment 3), we trained GPT-style decoders (1, 2, 4 or 6 layers; 3.17M parameters at 4 layers) on in-context group-table sequences laid out like the stimuli, with fresh symbols per table. A mixed-environment diet of groups, identity-free Latin squares and non-associative loops (A1, A2, A4, A6, by layer count) makes the history informative about the environment, and three 4-layer controls use groups only (G4), a fixed pool of labelled group tables (P4) or an exemplar diet without groups (I4). Predicted classes, validation criteria and a learning-failure rule were frozen before training (@app:trained), and each network was scored on the identical Experiment-1 trials, with words mapped to fresh random symbols per table (four draws).

### Positive control {#sec:ica}

Experiment 6 (*preregistered*) applied the fingerprint to the released In-Context Algebra model [@todd2026ica], a real network that should not be class R: a 4-layer, 8-head transformer with d=1024 and 16 symbol tokens, trained on in-context cyclic and dihedral group tables (C3–C10, D2–D5). We re-implemented it from the released code without executing that code and validated the port on the model's own task (@app:ica). Only cells with at most 16 distinct words fit the model, which excludes every t ≥ 2 and scramble cell, so the primary analysis is rule steps 1–3 on base t=0, with six directional predictions; interleaved fact order and six training checkpoints are *exploratory*, and a mechanism split of G cells against an exact loop-level baseline is *post hoc*.

### Statistics, preregistration and audits {#sec:stats}

Accuracies carry Wilson 95% intervals, except the rule's $\mathrm{acc}_L$ intervals, which are trial-bootstrap percentile intervals. Contrasts carry trial-level bootstrap 95% intervals (2,000 resamples within each cell), with seeds fixed per test in the main analyses; only the Experiment 2 per-model t4 − t0 CIs and the scoring re-check depend on processing order.

We apply no correction for multiple comparisons in the main analyses; a post-hoc Bonferroni check is in @app:mult. Of the 20 model-by-order $t = 0$ to $t = 4$ contrasts of the pretrained models, 6 have uncorrected $p < 0.05$ (2 increases, 4 decreases), and only Qwen3-0.6B at $n = 8$ (+0.031, $p$ = 0.0003) survives Bonferroni; OLMo-2-1B's +0.042 at $n = 6$ ($p$ = 0.0103) does not. For the null contrasts, correction widens the intervals: the largest upper bound of $t = 4$ minus scramble at $n = 6$ rises from +0.056 to +0.069 (Qwen3-4B-Base).

Experiments 1, 2, 3, 5 and 6 were *preregistered* (2026-10-06 to 2026-10-11), each before its outputs, except that Experiment 5's file was finalised minutes after a discarded six-trial smoke test; Experiments 4 and 5b are *exploratory*. We report every model and run, including nulls and interrupted runs. Two adversarial audits and later verification passes, automated self-checks run by the AI system that ran the analyses (see Statements), re-derived the Experiment 1–6 results, and their corrections are incorporated (@app:impl). @app:deviations lists all 23 deviations, including 100 rather than 200 trials per cell; apart from the ≥30-query guard, none altered a stimulus, threshold or decision rule.
