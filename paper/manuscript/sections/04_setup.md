## Experimental setup {#sec:setup}

All model inference and training ran on one shared laptop RTX 5070 GPU (8 GB); the exact learners and analyses ran on the CPU.

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

Prompts are the raw context plus the query, without a chat template. Queries were batched on copies of each trial's prefix cache rather than scored singly as preregistered; on the Experiment-1 models the two agree on 94–98% of argmax decisions, differing only on near-ties. For the 7B and 8B models, embeddings and the float32 output head ran on the CPU; in two checks this offload agreed with standard scoring on 91.9% and 94.7% of argmax decisions. Some result files mix revisions of the scoring script (@app:deviations). Re-scoring the four n=6 cells of Qwen3-1.7B-Base's primary contrast with the final code, one query at a time, moved per-cell G accuracy by at most 0.005 and left t4 − scramble at −0.006 [−0.034, +0.021] (originally −0.007 [−0.037, +0.020]; @app:impl).

nf4 quantisation [@dettmers2023qlora] was needed to fit the 4B, 7B and 8B models. A *preregistered* bf16-versus-nf4 control on Qwen3-1.7B-Base required an unchanged class and every per-cell G accuracy within 0.03. **The control failed.** nf4 lowered G accuracy in all 17 cells that contain G queries, by 0.020–0.067, and per-query argmax agreement was 56%, although the class (R at n=6 and n=8) and the null t4 − scramble contrast were unchanged (@sec:r-llm). As preregistered, all nf4 results, marked †, carry the caveat that G accuracy is biased low by about 0.04 and per-query predictions are unstable. The control shows that nf4 did not create an effect, not that it could not hide a weak one.

### Calibration {#sec:calib}

Because compounding presupposes Latin inference, Experiment 2a swept cue density (n=6, base histories, t ∈ {0, 4}, k ∈ {6, 10, 16, 22, 28}, 50 new trials per (t, k)) on eight models. The Latin threshold $k_L$ is the smallest k at which L accuracy, pooled over t, reaches 0.5; a model without one is *non-inferential* on this format.

A guard requiring at least 30 pooled L queries per k, added after the Qwen3-0.6B-Base sweep because k=6 has only 4 L cells in 100 trials, changes the status of two models. Without it, Qwen3-0.6B-Base and Qwen3-1.7B-Base answer 2 of 4 L queries at k=6, so $k_L = 6$; being R at k=10, they trigger the preregistered conditional step (addendum 2d), the only adaptive step allowed, which regenerates the n=6 main contrasts at k=16 (100 trials per cell, new seeds) and takes $\mathrm{acc}_L$ from k=16 base t=0. We ran it as specified after the final audit and report both readings (@sec:r-cal).

Two *exploratory* variants are not used for classification: an instructed task header (k ∈ {10, 22}) and a sorted format that lists the target's facts by row (k ∈ {16, 22, 28}), added after the first three ≤1.7B models showed chance-level L accuracy. As *post hoc* references we computed learner C and one-pass row/column elimination, which guesses uniformly among the values absent from the cell's row and column.

### Transformers trained from scratch {#sec:trained}

To test whether the fingerprint recovers the class implied by a known training diet (Experiment 3), we trained GPT-style decoders (pre-LN, rotary embeddings [@su2021rope], width 256, 8 heads, 1, 2, 4 or 6 layers; 3.17M parameters at 4 layers) on in-context group-table sequences laid out like the stimuli, in a 70-token vocabulary with fresh symbols per table and loss on answer tokens only. Diets use orders {4, 5, 6, 7, 8, 10} and exclude S3, D4 and Q8. The mixed-environment diet (A1, A2, A4, A6, by layer count) draws each sequence from groups (0.5), identity-free Latin squares (0.25) or non-associative loops (0.25), so that history is informative about the environment. Three 4-layer controls use groups only (G4), groups from a fixed pool of 32 labelled tables per order (P4), and an exemplar diet of isomorphic-copy tasks and identity-free Latin squares with no group or identity (I4).

Predicted classes were frozen before training (@sec:r-trained). The method counts as validated if A4 is A or B-type, I4 is N_iso, and P4 and A1 are R or N; a run whose in-training remaining-fact accuracy at n=6 never exceeds chance + 0.10 is *not learned* and excluded from validation.

Training ran for 14,000 AdamW steps [@loshchilov2019adamw] (β = 0.9/0.95, weight decay 0.1, clipping 1.0, seed 0) of $\lfloor 65536/L \rfloor$ sequences of length L, each step from a single (n, t), with 500 warmup steps and cosine decay to 10% of a $6 \times 10^{-4}$ peak ($10^{-3}$ for 1–2 layers); t ∈ 0–4 (t ≤ 2 at n=10, so that sequences fit 2,400 tokens). The target table is complete in training and the loss covers every answer token after the sequence's first fact, so the generator's cue size k, uniform in $[\lceil 0.15 n^{2} \rceil, \lfloor 0.45 n^{2} \rfloor]$, does not affect training. Four runs were interrupted by laptop standby or a scheduler change; three were resumed from their last checkpoint and one retrained from scratch, all reaching the preregistered 14,000 steps on a continuous learning-rate schedule (@app:deviations).

Each network was scored on the identical Experiment-1 trials, with words mapped to fresh random symbols per table (four draws), and classified by the unchanged rule; a *post hoc* diagnostic compares its predictions with learner C and cell-level row/column elimination.

### Reasoning probe {#sec:probe}

Experiment 4 (*exploratory*, not preregistered) asked whether direct answering limits Latin inference. Qwen3-1.7B answered one Latin-forced cell per trial under a chat prompt stating the Latin-square rule, so the probe measures inference given the rule. Decoding was greedy, with thinking on (at most 4,096 tokens) or off (at most 400); truncated thinking counts as incorrect. The planned design (four conditions × 30 trials, 1,536 thinking tokens) truncated all 13 thinking answers in its first condition, so we reduced it to 15 trials each of sorted and raw k=28.

### The CR axis {#sec:cr}

Experiment 5 (*preregistered*) tested the "compute versus retrieve" activation detector from our earlier preprint [@ramdan2026cr], whose stated limitation is that its compute prompts are longer and more complex than its retrieval prompts. Our stimuli remove this confound: in the n=6 leak cell at t=1 every query is retrievable, while in base at t=1 the same format, token count and query position require inference.

We ran the published detector code read-only, in bf16 rather than fp16: a mean-difference axis from the paper's 30 + 30 calibration prompts at the best of four layers, read at the last token of chat-template (primary) and raw inputs. The three bf16 models received all 26 queries per trial and the two nf4 models (†) 11–12 class-stratified queries, over 100 trials per condition. On Qwen3-0.6B-Base (bf16, uncached forwards) our forward pass reproduced the detector's hidden states exactly; the nf4 and CPU-embedding paths were not checked this way, and the cached path matched uncached forwards in an audit check (r ≥ 0.996; @app:impl). H1 predicts that base projects higher, toward "compute", than leak; it is supported if the trial-bootstrap CI excludes 0 in that direction in at least 3 of 5 models. H2 predicts a rise in the order L < O < G within base. The hypotheses were finalised minutes after a discarded 6-trial smoke test whose estimate pointed against H1; the direction was kept.

Experiment 5b (*exploratory*) followed the H1 result and a 20-trial audit pilot in which relabelling closed 53% and 71% of the gap, so it re-estimates a seen effect rather than testing it independently. Leak and base are matched in length and format but not vocabulary: a leak context has 6 distinct words, a base context 12 (@app:cr). **baseRel** writes the base history in the target's words (shared vocabulary, still non-retrievable); **leakRel** permutes the leak history's words by a random derangement (identical vocabulary, inconsistent answers). We report $f_{\mathrm{vocab}} = (\mathrm{baseRel} - \mathrm{base})/(\mathrm{leak} - \mathrm{base})$, the share of the gap reproduced by vocabulary alone, and leakRel − leak, retrievability with vocabulary fixed (predicted > 0), from uncached forwards on all 26 queries and 100 trials in four models.

### Positive control {#sec:ica}

Experiment 6 (*preregistered*) applied the fingerprint to the released In-Context Algebra model [@todd2026ica], a real network that should not be class R: a 4-layer, 8-head transformer with d=1024 and 16 symbol tokens, trained on in-context cyclic and dihedral group tables (C3–C10, D2–D5). We re-implemented it from the released code without executing that code.

On its own distribution the port reaches 0.973 held-out final-fact accuracy, and 0.959 under the upstream protocol on 1,024 fresh sequences, against the authors' 99.4%. Theirs is the best of about 290 training-time evaluations on one cached 512-sequence set; on our fresh sequences the released checkpoints from 20k steps on score 0.959–0.975, so the gap is consistent with checkpoint selection rather than a port error. Training loss is 0.917 (upstream: about 0.93), and an independent second forward pass agrees with the adapter on 3,200 of 3,200 answer positions (@app:ica).

The adapter maps each trial's words to distinct random symbols (four draws; no group identity maps to '0', as in training) and scores each query by forced choice over the target's n symbols. Only cells with at most 16 distinct words fit, which excludes every t ≥ 2 and scramble cell, so comp, held and w cannot be computed. The primary analysis is therefore rule steps 1–3 on base t=0, returning R, B-type or "non-R, non-B", with six directional predictions (@sec:r-ica). Interleaved fact order and six training checkpoints are *exploratory*; a mechanism split of G cells, compared with an exact loop-level baseline, is *post hoc*.

### Statistics, preregistration and audits {#sec:stats}

Accuracies carry Wilson 95% intervals, except the rule's $\mathrm{acc}_L$ intervals, which are trial-bootstrap percentile intervals. Contrasts carry trial-level bootstrap 95% intervals (2,000 resamples within each cell), with seeds fixed per test in the main analyses; only the Experiment 2 per-model t4 − t0 CIs and the scoring re-check depend on processing order.

Experiments 1, 2, 3, 5 and 6 were *preregistered* (2026-10-06 to 2026-10-11), each before its outputs, except that Experiment 5's file was finalised minutes after a discarded six-trial smoke test; Experiments 4 and 5b are *exploratory*. We report every model and run, including nulls and interrupted runs. Two adversarial audits re-derived the Experiment 1–5 numbers that existed at the time; later verification passes (2026-10-09 to 2026-10-11) re-derived the results produced after them (learner C, the count-shortcut diagnostic, the scoring checks, Experiment 5b and the addendum-2d run) and those of Experiment 6. All corrections are incorporated. @app:deviations lists all 23 deviations, including 100 rather than 200 trials per cell. Apart from the ≥30-query guard, none altered a stimulus, threshold or decision rule.
