# Methods

## Task
A trial presents one or more operation tables as fact lists `x * y = z`, one fact per line in random order, over words that are single tokens. Each table is preceded by a neutral header (`Table A:`, `Table B:`, …).

- The first *t* tables (t ∈ {0, 1, 2, 4}) are **history** tables. They are complete, and each uses a fresh set of words: no word is reused across the tables of a trial.
- The last table is the **target**. It reveals *k* of its n² cells, the **cue**.
- Every unrevealed target cell is then queried independently with `\nwa * wb =`. The answer is a forced choice among the target's n words, scored by the model's next-token log-probability of `" w"`.

Stimuli were produced by a single generator (`runs/c10/code/stimuli.py`). It applies these controls:
- fresh words per table (except in the leak control, which repeats the target's labelled table by design), with a random word↔element bijection per table;
- random fact order;
- a blacklist of numerals, ordinals and identity-suggestive words;
- a vocabulary of 101 words that are single tokens, with and without a leading space, in every tokenizer used.

## Forcing classes
Every queried cell is labelled exactly with the weakest axiom class under which the cue determines it. Latin classes use enumeration (n ≤ 5) or SAT; loop classes use enumeration (n ≤ 6) or SAT; group classes use enumeration of all labelled tables of every group of order n:
- **L**: every Latin square consistent with the cue agrees on the cell.
- **O**: every loop agrees (identity and cancellation).
- **G**: every group table on the same labels agrees. These cells are forced *only* through associativity.
- **Ab / Cy**: forced by commutativity or cyclicity.
- **U**: undetermined.

Accuracy on G cells is the primary measure, because only a learner that uses group structure can exceed the loop level on them.

## Conditions
| condition | history | target |
|---|---|---|
| base | random groups of order n | random group |
| scramble | Latin squares with no two-sided identity, matched in fact count and token length (6 of 400 n=6 tables have a one-sided identity) | random group |
| heldout | n=6: Z6 only; n=8: abelian types only (Z8, Z4×Z2, Z2³) | an unseen type (n=6: S3; n=8: D4 or Q8) |
| leak (recall positive control) | the last history table *is* the target's own labelled table | random group |
| loops (n=5) | group-like non-associative loops | Z5, or a loop |

**Main design (Experiments 1–2).** n=6 (k=10) and n=8 (k=16) cover base t ∈ {0, 1, 4}, scramble t=4 and heldout t ∈ {1, 4}. n=6 alone adds base t=2 and leak t=1. n=5 (k=6) covers base and loops. There are 100 trials per cell: 1,800 trials and 57,200 scored cells per model.

This design deviates from its source (c10) in four ways:
- 100 rather than 200 trials per cell;
- the n=8 cells were promoted to a secondary analysis;
- the abc, lesion and fine-tuning cells were deferred;
- Experiment 1 used the models available locally (≤1.7B), and Experiment 2 extended the set to 8B.

**Calibration (Experiment 2a).** n=6, t ∈ {0, 4}, k ∈ {6, 10, 16, 22, 28}, with 50 trials per (t, k). Two variants:
- an instructed variant (task header) at k ∈ {10, 22};
- a sorted variant (target facts grouped by row) at k ∈ {16, 22, 28}. This variant was added after the ≤1.7B sweeps (see Deviations).

## Idealised learners
Seven exact algorithms are scored on the identical trials (`exp1/ideal/`):
- **R:** recall only.
- **N:** uniform over Latin-feasible values.
- **N_iso:** exemplar matching up to relabelling against the history tables, with Latin fallback.
- **A:** induces axioms (identity, associativity, commutativity, cyclicity) from the history tables, then applies them.
- **Asl:** like A, but infers "group" from an identity alone.
- **B:** Bayes-optimal with the true environment prior.
- **Bh:** hierarchical Bayes over the Latin, loop, group, abelian and cyclic environments.

Ties are broken uniformly at random with a fixed seed.

An eighth learner was added post hoc (2026-10-09), after the trained networks were found to use it (Results R4). **C (count shortcut)** answers uniformly among the target words that occur least often as a result in the revealed cue (`exp1/ideal/code/ideal_count.py`). Agreement with the stimulus labels was checked: cell classes had 0 mismatches over 57,200 cells, and SAT agreed with exact counts on 913 trials.

## Decision rule (preregistered, frozen before any model output)
1. acc_L is the L-cell accuracy at base t=0. If acc_L < 1/n + 0.25, the model is **R**.
2. Accuracies are rescaled: g(x) = (x − 1/n)/(acc_L − 1/n).
3. If g(G at t=0) ≥ 0.85, the model is **B-type** (the prior is already in place).
4. base = max(g(G at t0), g(G at scramble t4)), and comp = g(G at t4) − base. If comp < 0.15, the model is **N**.
5. held = g(G at heldout t4) − base. If held ≥ 0.6·comp, the model is **A**; otherwise **N_iso**.

**Validation.** Applied to the idealised learners on the identical trials, the rule returns each learner's own class at both n=6 and n=8, on both accuracy and mean p_gold, with 0 misclassifications. Asl is classified A.

## Models (Experiments 1–2)
All checkpoints are ungated and Apache-2.0 licensed:

| model | precision |
|---|---|
| Qwen3-0.6B, Qwen3-1.7B (post-trained) | bf16 |
| SmolLM2-1.7B-Instruct (Llama architecture) | bf16 |
| Qwen3-0.6B-Base, Qwen3-1.7B-Base | bf16 |
| Qwen3-4B-Base | nf4 |
| Qwen3-8B-Base | nf4 |
| OLMo-2-0425-1B | bf16 |
| OLMo-2-1124-7B | nf4 |

nf4 quantisation was needed to fit the 8 GB GPU. A bf16-versus-nf4 control on Qwen3-1.7B-Base was preregistered: the class unchanged, and every per-cell G accuracy within 0.03.

**The control failed.** nf4 lowered G accuracy in all 17 cells that contain G queries, by 0.020–0.067, and per-query argmax agreement with bf16 was 56%. The class (R at n=6 and n=8) and the null t4−scramble contrast were unchanged.

As preregistered, every nf4 result carries this caveat (†): Qwen3-4B-Base, Qwen3-8B-Base, OLMo-2-7B, and the Experiment-5 4B and 8B projections. G accuracy is biased low by about 0.04, and per-query predictions are unstable.

## Implementation and hardware
All model inference and training ran on one laptop RTX 5070 GPU (8 GB), shared with other applications. The exact learners and all analyses ran on the CPU.

During unattended runs the laptop entered Modern Standby several times. Windows Smart App Control blocked Python extension DLLs after a resume twice:
- `_multiprocessing` on 2026-10-08 at 07:38, which failed every queued job until restart;
- SciPy's `_hausdorff` at 16:45.

SciPy was removed from the GPU environment, and no scoring, training or analysis code imports it. Scoring jobs resume at trial level and training at checkpoint level, except the Experiment-5 projections (cr_probe.py), which restart a model from scratch (none was interrupted). Interrupted jobs were rerun from where they stopped.

- **Query scoring.** Each trial's prefix is encoded once.
  - Queries with equal suffix length are scored in chunks of up to B on a copy of the prefix KV cache.
  - A one-query chunk is scored directly on the prefix cache, which is then cropped back. The arithmetic is identical, with one fewer copy.
  - B is set per trial as min(batch_max, KV budget / prefix KV size) and halved after an out-of-memory error. It therefore varies with context length, and so with condition (e.g. OLMo-2-7B: B=8 at t=0, B=1 at t=4).
  - The scoring script was revised at least three times during Experiment 2 (2026-10-07):
    - at 13:16 it gained the memory cap, OOM back-off and CPU offload;
    - between 17:43 and 21:21 it gained per-trial provenance fields;
    - at 21:25 it gained the in-place single-query path, after the first OLMo-2-7B attempt ran out of memory.
  - Some files mix versions:
    - Qwen3-1.7B-Base bf16: the n=5 cells, the n=6 base and heldout cells and n6_leak_t1 trials 0–38 (1,039 trials) were scored with the first version; n6_leak_t1 trials 39–99, scramble t=4 and the n=8 cells (761 trials) with the last.
    - Qwen3-4B-Base main: the first 1,584 trials, up to n8_heldout_t1 trial 83, were scored with the first version, and the remaining 216 with an unarchived intermediate version. Its calibration file used the first version only.
    - OLMo-2-7B calibration: the 50 kept trials (ki_n6_k10 trials 0–49; batch_max 16, 60% cap) come from the unarchived version in use before 21:25.
  - The intermediate versions that scored Qwen3-8B-Base, the 4B remainder and those 50 OLMo-2-7B trials were not archived.
  - Batching changes bf16 results slightly. On the Experiment-1 models, batched and sequential scoring agreed on 94–98% of argmax decisions (max |Δp_gold| 0.05, near-ties).
  - To check that the version mix does not matter, the four n=6 cells entering Qwen3-1.7B-Base's primary contrast were re-scored with the final code and B=1 (`exp2/offchk_result.md`). Per-cell G accuracy moved by at most 0.005, argmax agreement was 94–98%, and t4 − scramble was −0.006 [−0.034, +0.021] against −0.007 [−0.037, +0.020] originally.
- **7–8B models (Qwen3-8B-Base, OLMo-2-7B).**
  - On the GPU: the nf4 transformer layers, final norm and rotary embedding.
  - On the CPU: token embeddings, looked up from the bf16 checkpoint weights, and the output projection, computed in float32 from the checkpoint's lm_head. All other models compute logits on the GPU in bf16.
  - Archived re-check on Qwen3-1.7B-Base, batch 1, 72 trials: offload vs standard scoring agreed on 2,102 of 2,288 argmax decisions (91.9%), with mean |Δp_gold| 0.007 (max 0.05) and accuracy 0.199 vs 0.201. On Qwen3-4B-Base nf4, which is the 8B configuration, nf4 + offload vs plain nf4 agreed on 1,083 of 1,144 decisions (94.7%), with mean |Δp_gold| 0.004 (max 0.02) and accuracy 0.199 vs 0.197.
  - An earlier, unarchived development check had given 139/148.
- **Memory.** From 2026-10-07 13:16, each scoring process was capped at 60% of GPU memory (62% for OLMo-2-7B), with OOM batch back-off. Earlier runs had neither. The first OLMo-2-7B calibration attempt ran out of memory after 50 trials and was resumed at trial level with batch_max 8; those 50 trials were kept.

## Controlled transformers (Experiment 3)
GPT-style decoders are trained from scratch: pre-LN, RoPE, d_model 256, 8 heads, 1–6 layers. The training data are in-context group-table sequences in a 70-token vocabulary, using the same layout as the stimuli: fresh symbols per table, and loss on answer tokens only.

- **Training diet.** Orders {4, 5, 6, 7, 8, 10}, with S3, D4 and Q8 never seen in training.
  - The mixed-environment diet draws each sequence from groups (0.5), identity-free Latin squares (0.25) or non-associative loops (0.25). Earlier tables then carry information about the environment.
  - Controls:
    - groups only (G4);
    - groups only, drawn from a fixed pool of 32 labelled symbol tables per order (P4);
    - an exemplar diet (I4). Half of I4's sequences are isomorphic-copy tasks (identity-free Latin history; the target is a relabelled copy of one history table), and half are identity-free Latin squares. The diet never contains a group or an identity.
- **Schedule.** 14,000 AdamW steps (β = 0.9/0.95; weight decay 0.1 on matrices; clip 1.0; bf16 autocast; seed 0).
  - Each step holds ⌊65,536/L⌋ sequences of a single (n, t).
  - The learning rate warms up over 500 steps, then follows a cosine decay to 10% of a 6e-4 peak (1e-3 for 1–2 layers).
  - k is uniform in [⌈0.15n²⌉, ⌊0.45n²⌋], t is in 0–4, and t ≤ 2 at n=10.
- **Interruptions.** Unattended runs were stopped four times. Three (A2, G4, A6) were stopped by the wall-clock safety cap after the laptop entered standby. A4's first attempt (launched with a 240-min cap) ended at about step 3,200 without reaching its cap or saving a checkpoint, when the concurrent GPU queues were replaced by the sequential scheduler (10-07, about 14:19). A4, G4 and A6 were completed to the preregistered 14,000 steps from their last checkpoint, with optimiser state restored and a fresh data-stream seed (seed + step). A2 was retrained from scratch:
  - A4 was interrupted at about 3,200 steps and resumed from its step-2,000 checkpoint, so its log has 24 duplicated steps and its eval.csv has two step-3,000 blocks.
  - G4 stopped at step 9,934 and A6 at step 13,471. Both were scored at that point (`exp3/res/_G4_step9934.jsonl`, `_A6_step13471.jsonl`; excluded from analysis), then resumed on 2026-10-09 and re-scored at 14,000.
  - A2 stopped at step 1,759 after a stall and was retrained from scratch with the same seed. The retrain reproduces the first run's trajectory (loss at step 1,750: 2.1442 vs 2.1461).

  The learning-rate schedule is continuous across every resume: every logged learning rate matches the planned schedule, and there is no loss discontinuity. On 2026-10-09 at 08:11 the safety-cap code was changed to count active compute time only, which applied to the G4 continuation (the A6 continuation ran without a cap). A6's first run, launched at 07:46 that day, still used the wall-clock cap, which fired at step 13,471 after a standby gap. These continuations were not among the preregistered allowed adaptations, but they only complete the preregistered schedule.
- **Evaluation.** Each trained model is scored on the *identical* Experiment-1 trials, with words mapped to fresh random symbols per table (four symbol draws), and classified by the unchanged decision rule. Predictions for each run were frozen in `exp3/PREREGISTRATION.md` before training.
- **Post-hoc diagnostic** (`exp3/heur/`). For each trial we record:
  - within-trial prediction constancy;
  - the share of predictions in the least-revealed set;
  - the accuracy of learner C, and of cell-level row/column elimination.

  The last two are also recomputed on the in-training evaluation conditions with the training generator.

## Reasoning probe (Experiment 4; exploratory, not preregistered)
Qwen3-1.7B (post-trained, bf16) answered one Latin-forced cell per trial, from Experiment-2 t=0 calibration trials. A chat prompt stated the Latin-square rule. Decoding was greedy, with thinking on (≤4,096 new tokens) and off (≤400). Truncated thinking counts as incorrect. Non-thinking answers without an ANSWER line (6 of 30, all cut off at 400 tokens) were scored by the last candidate word mentioned; all six were incorrect.

The planned design was four conditions × 30 trials with a 1,536-token budget. All 13 thinking answers in the first condition were truncated, so it was reduced to 15 trials each of sorted and raw k=28 (`exp4/NOTE.md`, amendment 2026-10-08 20:52).

## CR-axis projections (Experiment 5; preregistered 2026-10-08)
- **Detector.** The published compute–retrieve detector code was used read-only, in bf16 instead of the paper's fp16: a mean-difference axis on the paper's 30 + 30 calibration prompts, at the best of four candidate layers by AUC.
- **Contrast.** n6_base_t1 versus n6_leak_t1, with hidden states taken at the last token from decoder-layer outputs, in detector (chat-template) and raw formats.
- **Models.**
  - Qwen3-0.6B-Base, 1.7B-Base and 1.7B (bf16), all 26 queries per trial.
  - Qwen3-4B-Base and 8B-Base (nf4†), 11–12 class-stratified queries per trial (the stratified rounding gives 11 in 20 of 400 trials; 12 were preregistered).
  - There were 100 trials per condition, with a shared-prefix KV cache.
- **Numerical checks.**
  - On Qwen3-0.6B-Base (bf16, 3 prompts), calling the decoder stack directly reproduced `CRDetector._get_hidden` exactly (max relative difference 0) on uncached forwards. The nf4 and CPU-embedding paths were not checked this way.
  - The cached path was checked in the audit against full uncached forwards: r ≥ 0.996, median relative difference ≤ 3.5%, and sample condition differences reproduced within 1.3 units.
- **Decision rule.** H1 is supported if base > leak, with a trial-bootstrap CI excluding 0, in ≥3 of 5 models.
- **Timing.** The hypotheses were finalised minutes after a 6-trial CPU smoke test, whose outputs were discarded and whose estimate pointed against H1.
- **Experiment 5b (exploratory, written before its full run; `exp5/CONTROL_vocab.md`).** This control relabels the history table so that a non-retrievable history uses the target's vocabulary (baseRel), or so that a retrievable history becomes inconsistent (leakRel).
  - It uses uncached forwards, all 26 queries and 100 trials, on four models (8B was not run).
  - Qwen3-0.6B-Base ran one query per forward. The other three models ran the 26 queries of a trial-condition in one right-padded batch.
  - All rows in a batch have equal token length, so no padding occurs. A 3-trial check reproduced the per-query projections exactly.

## Positive control (Experiment 6; preregistered 2026-10-11)
- **Model.** The released In-Context Algebra checkpoint (`algebra_gpt_best.pt`, sha256 9c0ee347…; https://algebra.baulab.info/weights/): 4 layers, 8 heads, d=1024, RoPE, 16 symbol tokens. It was trained on mixed in-context cyclic and dihedral group tables, C3–C10 and D2–D5.
- **Re-implementation.** The model was re-implemented from the released code (GitHub ericwtodd/algebra at 532a6a2) without executing it, and loaded with `torch.load(weights_only=True)`.
- **Validation on its own distribution.**
  - Held-out final-fact accuracy is 0.973 at 200 facts.
  - It is 0.959 under the upstream protocol on 1,024 fresh sequences, against the authors' 99.4%. Theirs is the best of about 290 training-time evaluations on one cached 512-sequence set; on our fresh sequences the released checkpoints score 0.959–0.975 (`exp6/validate_ckpts.json`).
  - Full-sequence training loss is 0.917 (upstream curve: about 0.93).
  - A second, independent forward pass agrees with the adapter on 3,200 of 3,200 answer positions.
- **Adapter.**
  - Each trial's words get distinct random symbols (4 draws), redrawn so that no group identity maps to '0', as in the model's training.
  - Facts are emitted as `,ab=c` in stimulus order, without table headers.
  - Each query `,ab=` is scored by forced choice over the target's n symbols.
- **Cells.** Only cells with ≤16 distinct words fit: n5/n6/n8 base t0, n6/n8 base t1, n6/n8 heldout t1, n6 leak t1, the n=6 t=0 cue sweep (raw and sorted), and the addendum-2d k=16 t ≤ 1 cells.
- **Primary analysis.** Decision-rule steps 1–3 on base t0, since steps 4–5 need t=4 and scramble cells.
- **Exploratory.** Interleaved fact order, and six released training checkpoints (2 draws).
- **Post hoc.** A mechanism split of G cells by target commutativity and by whether the transposed cell is revealed. It is compared with an exact loop-level baseline (uniform over loop-feasible values, from the stimulus generator's SAT engine; `exp6/loop_level.py`).
- **Environment.** Smart App Control blocked the GPU environment's venv launcher on 2026-10-11, so these runs used the base CPython 3.12 interpreter with the same site-packages (torch 2.7.1+cu128).

## Statistics
- Accuracies carry Wilson 95% intervals over cells, except the acc_L intervals reported by the decision rule, which are trial-bootstrap percentile intervals.
- Contrasts carry trial-level bootstrap 95% intervals (2,000 resamples, resampling trials within each cell).
- Bootstrap seeds are fixed per test in the main analyses (collect.py, cal_summary.py, exp1/analyze.py, analyze5.py, analyze_ctrl.py), so those CIs are reproducible and independent of processing order. The per-model t=4 − t=0 CIs in exp2/RESULTS2_tables.md and the two CIs in exp2/offchk_result.md use a shared generator: they are reproducible for a fixed set of result files, but they depend on processing order.
- Every model and run is reported, including nulls and runs that were interrupted and completed (see Deviations).

## Deviations from preregistration
Times are local.

| # | prereg | planned | done | when |
|---|---|---|---|---|
| 1 | Exp1 | 7–8B models | Qwen3-0.6B/1.7B and SmolLM2-1.7B; extended to 8B in Exp2 | 10-06 |
| 2 | Exp1 | 200 trials per cell (c10) | 100 | 10-06 |
| 3 | Exp1 | n=8 optional | promoted to secondary; abc, lesion and fine-tuning cells deferred | 10-06 |
| 4 | Exp2 2a | k_L rule | guard of ≥30 L queries added after the Qwen3-0.6B-Base sweep. Contrary to the 10:16 log entry, it *does* change that model's status, and Qwen3-1.7B-Base's: without it, k=6 gives 2/4 = 0.50, so k_L = 6 and 2d applies. 2d was run on 10-09 (both remain R at k=16, with acc_L 0.18 and 0.22) | 10-07 10:16 |
| 5 | Exp2 | no sorted variant | kso_ stimuli generated at 10:32:57, 22 s after Qwen3-1.7B-Base calibration scoring began and before any of its output was read (logged 10:36). The kso rows of Qwen3-0.6B-Base, Qwen3-1.7B, SmolLM2-1.7B and Qwen3-1.7B-Base, which were calibrated before the kso files existed, were scored on 10-09 with the final code | 10-07 / 10-09 |
| 6 | Exp2 2c | tolerance ≤0.03 | FAILED (max \|ΔG\| 0.067; argmax agreement 56%) | 10-08 ~17:16 |
| 7 | Exp2 2b | 8B and 7B in nf4 | also CPU embeddings and an fp32 CPU head | 10-07 / 10-08 |
| 8 | Exp2 | a single scoring procedure | run.py revised at least three times; some result files mix versions; the 8B file, the last 216 trials of the 4B main file and the first 50 OLMo-2-7B calibration trials were scored with unarchived versions | 10-07 13:16, 17:43–21:21, 21:25 |
| 9 | Exp2 2d | conditional k=16 | not run at the time (excluded by the guard of row 4); run on 10-09 for Qwen3-0.6B-Base and Qwen3-1.7B-Base with new-seed stimuli (`exp2/make_stim_k16.py`) | 10-09 |
| 10 | Exp3 | 14k steps, one run | A4 interrupted at about 3,200 steps and resumed from step 2,000 | 10-07 13:26–14:29 |
| 11 | Exp3 | as above | A2 stopped by its cap at step 1,759 after standby; retrained from scratch with the same seed | 10-07 22:52 to 10-08 21:22 |
| 12 | Exp3 | as above | G4 stopped by its cap at step 9,934 during standby; scored, then resumed to 14,000 | 10-08 to 10-09 16:57 |
| 13 | Exp3 | as above | A6 stopped by its cap at step 13,471; scored, then resumed to 14,000 | 10-09 11:49–17:04 |
| 14 | Exp3 | `--minutes` cap 110/150 (README) | 150, 60, 90 or 240, later 90, finally none; from 10-09 the cap counts active time only | various |
| 15 | Exp3 | learning-failure rule; decision criteria | applied in Results R4 (after the second audit) | 10-09 |
| 16 | Exp3 | C4 optional | not run | allowed |
| 17 | Exp4 design note (exploratory, not preregistered) | 4 cells × 30 trials, 1,536 tokens | 2 cells × 15 trials, 4,096 tokens (chosen after seeing v1 truncations) | 10-08 20:52 |
| 18 | Exp5 | 12 queries per trial for 4B and 8B | 11 in 20 of 400 trials | 10-09 |
| 19 | Exp5 | forward-path equivalence | checked only for the uncached forward; the cached path was validated later by the audit | 10-09 |
| 20 | Exp5 | interpretations for "supported" and "not supported" | the observed outcome (significant reversal in 3 of 5 models) was not covered; it is reported as such | 10-09 |
| 21 | — | — | post-hoc additions: learner C, the count-shortcut diagnostic, the answer-bias diagnostic (exp1/diag.py), Experiment 5b | 10-07 / 10-09 |
| 22 | Exp1 (Exp2 by reference) | each query scored on the prefix cache, cropped back after each query | queries of equal suffix length batched (up to B per chunk) on copied prefix caches; 2–6% of near-tie argmax decisions differ from sequential scoring | 10-06 |
| 23 | Exp6 | adapter validation 'exactly' as the upstream protocol, expected ≥0.99; steps 1–3 bootstrap seeded as in analyze.py | validation used the same generator and final-token argmax but on 1,024 fresh sequences (not the cached 512-sequence set): 0.959, explained by checkpoint selection (validate_ckpts.json); the first analysis run seeded steps 1–3 differently and was corrected (CI endpoints moved ≤0.01); the mechanism split and loop-level baseline were added after the primary results; the B-type and unseen-type predictions were not met and are reported as such | 10-11 |

Further notes:
- The resume path overwrites `config.json` with the resume invocation's arguments. Resume arguments were checked by hand against the original runs, and they match on every setting that affects the schedule and data.
- P4's in-training `isocp6_t1` value (1.000) is an artefact: in pool mode the generator returns memorised pool tables. It is not used anywhere.
- Apart from the ≥30-L-query guard added to the k_L rule (row 4), none of these changes altered a stimulus, threshold or decision rule.
