# Experiment 3: small transformers trained from scratch on in-context group-table completion

Experiment 3 is a controlled positive control for the compounding fingerprint (c10 §5.4–5.5), and a test of the theory's bounded-inference predictions. Small GPT-style models are trained from scratch on sequences of operation tables. Each checkpoint is then scored on the **exact Experiment-1 stimulus files**, and the result goes through **Experiment 1's own `analyze.py`** (decision rule R / N / N_iso / A / B-type).

Status (2026-10-07): the code is built and tested on CPU, and a short GPU throughput test was run. No real training run has been made yet. See "What was tested" and "GPU throughput". (Superseded: see "Run history and outcome" at the end.)

## Files

| file | what it does |
|---|---|
| `tables.py` | Verbatim copy of `runs/c10/code/tables.py`. A test checks it is identical to c10 for n = 1..10. |
| `data.py` | Token vocabulary and sequence generator (`Cfg`, `Gen`), plus a decoder and type identifier used by the tests. |
| `model.py` | GPT decoder: pre-LN, RoPE or learned positions, tied embeddings, SDPA attention. It accepts custom position ids and an attention mask, which `eval_fp.py` uses. |
| `train.py` | AdamW, warmup and cosine schedule, bf16 autocast on CUDA, grad clipping, gradient accumulation, periodic held-out evaluation, checkpoint and resume, `--minutes` budget, CSV logs. |
| `eval_fp.py` | Scores a checkpoint on `exp1/stim/*.jsonl` and writes `res/<name>.jsonl` in the Exp-1 results format. |
| `analyze3.py` | Runs `exp1/analyze.py`, unmodified, on `exp3/res/`. Writes `exp3/RESULTS_tables.md` and `exp3/results_summary.json`. |
| `tests.py` | The CPU test suite. |

## Sequence format

Vocabulary: `PAD=0, BOS=1, '*'=2, '='=3, ';'=4 (end of fact), '|'=5 (table separator)`, then symbols `s0..s(S-1)`. The default is S = 64, so the vocabulary is 70 tokens.

```
BOS  a * b = c ; a * b = c ; ... | ... | <target: k cue facts> <remaining target facts>
```

- **Tables.** Each history table lists all n² facts in random order. The target is the last table. Its first k facts are the cue and the rest follow. Lengths: n=6, t=4 is 1,085 tokens. n=8, t=4 is 1,925 tokens. n=10, t=2 is 1,803 tokens.
- **Symbols.** Each table gets a fresh random injection from elements to symbols. No symbol is reused across the tables of a sequence. A random injection applied to the c10 base table makes every labelled table of a type equally likely.
- **Loss** (`--loss_on all`, as in Todd et al., arXiv 2512.16902). Next-token loss is taken only on answer tokens (the token after `=`), for every fact except the first fact of the sequence. `--loss_on target` restricts the loss to the target's facts after its k cues.
- **Batching.** Every batch holds a single (n, t), so there is zero padding and no packing across trials, which would leak extra "history" into a trial. (n, t) is drawn with probability proportional to π(n,t)·len, and B = tokens_per_step / len. Per-sequence frequencies therefore follow π: n uniform, then t uniform.

**Training-distribution options:**
- `--orders`: default 4 5 6 7 8. Orders 9 and 10 are available (Z9, Z3×Z3, Z10, D5). t is capped per n so that the sequence fits `--max_len`; with the default 2400, n=10 gets t ≤ 2.
- `--types`: `all | cyclic | abelian | noheld`. `noheld` means every group except HELDOUT = {S3@6, D4@8, Q8@8}, which is exactly the set of Exp-1 held-out targets.
- `--t_min/--t_max`: default 0..4.
- `--k_frac lo hi`: k is drawn uniformly in [⌈lo·n²⌉, ⌊hi·n²⌋]; default 0.15–0.45.
- `--pool_M M`: low task diversity. A fixed pool of M labelled symbol tables per order is drawn once from the seed, and every table of every sequence comes from the pool. *Deviation:* with S = 64, the tables of one sequence can share symbols in pool mode. They are chosen to minimise overlap; the mean overlap is about 1 symbol per sequence.
- `--env_mix`: added beyond the initial task specification (not included), for the reason given in the next section. Each sequence draws an environment, and its history and target both come from it:
  - `group`: groups of the allowed types;
  - `latin`: identity-free Latin squares, the same family as Exp-1's scramble control;
  - `loop`: non-associative loops at n ≥ 5. At n = 4 every loop is a group, so it falls back to `latin`;
  - `isocopy`: history tables are identity-free Latin squares, and the target is a relabelled copy of a random history table. It is an exemplar-matching task: it never shows a group or an identity element.

**Evaluation during training** (`eval.csv`, every `--eval_every` steps, 64 sequences per condition):
- Scoring is teacher-forced, forced choice over the target's n symbols, on the target facts after k = 10 (n=6) or k = 16 (n=8) cues.
- `acc` is the mean over all remaining facts. `q1` is the first fact after the cue only.
- Conditions:
  - `in{6,8}_t{0,4}`: training types;
  - `ho{6,8}_t{0,4}`: held-out target (S3; D4/Q8), with history from the training types;
  - `scr{6,8}_t4`: identity-free Latin history with a training-type target;
  - `isocp6_t1`: the isocopy task.

## Two design facts to know before choosing runs

1. **`noheld` training at n ≤ 8 contains no non-abelian group.** The only non-abelian groups of order ≤ 8 are S3, D4 and Q8, and all three are held out. With orders 4–8, `noheld` is therefore identical to `abelian`. A held-out test on S3/D4/Q8 then asks for transfer from abelian-only data. That is stricter than Todd et al., who trained on D3–D5.
   - To give the positive control any non-commutative experience, add order 10: `--orders 4 5 6 7 8 10` adds D5 (and Z10), with t ≤ 2 at n=10.
2. **Group-only training should not compound (c10 Prop C).** If every training sequence is a group, the history carries no information about the target's class. A model that fits that distribution has the group prior in its weights. Under Exp-1's rule it then shows up as **B-type** if it learned associative inference (g(G at t=0) ≥ 0.85), or as **N** (flat) if it did not.
   - Compounding needs an environment that is uncertain at t=0, so that history can tell the model "this is a group". That is the role of `--env_mix group=0.5,latin=0.25,loop=0.25`.
   - The grid below keeps the literal group-only run as the Prop-C control for (a), and uses the mixed environment as the positive control.

## How to run

GPU env: `C:\Users\%USERNAME%\AppData\Local\Temp\cgpu\Scripts\python.exe`. Run from `exp3/`. Check that `C:\Users\%USERNAME%\AppData\Local\Temp\cthk\GPU_BUSY` does not exist before you use the GPU.

```
# tests (CPU, about 1 min)
python tests.py

# train (default model: 4 layers, d_model 256, 8 heads, RoPE, 3.17M params; 65,536 tokens/step in micro-batches of 16,384)
python train.py --name A4 --orders 4 5 6 7 8 10 --types noheld --env_mix group=0.5,latin=0.25,loop=0.25 \
                --steps 14000 --minutes 80
python train.py --name A4 ... --resume          # continue from runs/A4/last.pt

# fingerprint on the exact Exp-1 stimuli (1800 trials, 57,200 queries), then Exp-1's analysis
python eval_fp.py --ckpt runs/A4/final.pt --name A4 [--draws 4]
C:\Users\%USERNAME%\AppData\Local\Temp\ccv\Scripts\python.exe analyze3.py     # or the GPU env python
```

**How `eval_fp.py` maps the stimuli:**
- Every distinct word of a trial gets a distinct random symbol, seeded by file, trial and draw. Exp-1 uses fresh words per table, so this gives fresh symbols per table. The leak control keeps its repeated words, and so its symbols.
- `Table X:` headers become `|`, with none before the first table.
- Each query `wa * wb =` is appended as `a * b =`.
- All queries of a trial are packed after a single copy of the context. Each query block sees only the context and itself, and re-uses positions L..L+3. A test checks this equals separate forwards: max |Δlogit| 2e-7.
- `p_gold` is the softmax over the n candidate symbols. `cand_mass` is the full-vocabulary probability on the candidates. `--draws D` averages over D symbol assignments.
- Output fields are the same as `exp1/run.py`'s.
- Files whose names start with `_` are skipped by `analyze.py`, as in Exp-1.

## What was tested (CPU, all passing: `python tests.py` → ALL PASSED)

1. **`test_tables_copy`.** `exp3/tables.py` gives the same base tables as c10's for n = 1..10. Each is Latin, associative and has an identity.
2. **`test_data_groups`.** 1,680 generated sequences were decoded back from tokens, covering every `types` mode, both loss modes and orders 4–10. Each one checks:
   - every table is a complete n² Latin table, and its isomorphism type, identified by invariants, matches the recorded type;
   - the symbol table equals the c10 base table under the recorded injection;
   - no symbol is shared across the tables of a sequence;
   - every loss label sits at a position whose input is `=` and whose target is that fact's answer, for all facts except the first (or, in `target` mode, exactly the target facts after its k cues);
   - target positions lie inside the last table;
   - type restrictions hold: `noheld` produces all and only the non-held-out types, `cyclic` only Z_n, `abelian` only commutative types;
   - k lies in range, and lengths fit max_len, including the t-cap at n = 10.
3. **`test_data_envs_pool`.**
   - In the mixed environments, history and target always share the environment. `latin` tables have no identity. `loop` tables have an identity and are non-associative.
   - Each `isocopy` target equals a history table up to relabelling.
   - The scramble-history evaluation override works.
   - In pool mode, every table is a pool entry.
4. **`test_eval_encoding`.** Five trials from each of the 18 Exp-1 files are encoded. The cue facts plus the gold answers rebuild a complete target of the declared type: a group type, or a non-associative loop for `loopsLT`. History tables have their declared types: group, `latin-noid` or loop.
   - Symbols are fresh per table, except that the leak table repeats the target's symbols.
   - The longest prefix is 1,637 tokens.
   - *Observed Exp-1 metadata quirk:* in `n6_leak_t1`, `history_types[-1]` records the originally drawn type (e.g. Z6), but the table actually shown is the target's own table (e.g. S3). This comes from `stimuli.py`'s leak branch. It does not affect Exp-1's analysis.
5. **`test_packed_equivalence`.** Packed queries match separate forwards, with RoPE and with learned positions.
6. **`test_overfit`.** A 2-layer, d=64 model on one fixed batch of 8 sequences goes from loss 4.27 to 0.0016, with label accuracy 1.000.
7. **`test_train_smoke`.** `train.py` runs end to end on CPU: mixed environment, logs, two evaluations, checkpoints. Resume restores the step.
8. **`test_eval_fp_end_to_end`.** 72 trials (4 per file) are scored, including the `--draws 2` path. Every record has exactly Exp-1's keys. `analyze.load` reads all 18 cells, and `analyze.decision` runs at n=6 and n=8 with bootstrap.
   - Separately, `analyze3.py` was run on a temporary copy of that file. It produced the Exp-1 tables and decisions, and the temporary outputs were then deleted. `exp1/` was not modified.

9. **Grid configs.** Each data config used in the grid (A, G, P, I and C, with orders 4–8 and 10) was built, and one batch per (n,t) was run through the same structural checks: 112 sequences per config. `Gen` initialisation takes up to 14 s for the mixed environments, which build their Latin and loop pools.

**Extra CPU sanity runs:**
- **Learning signal.** A 2-layer, d=64 model (0.1M params) was trained for 3,000 steps of 8k tokens on n = 4–5, taking 5.5 min (`runs/_cpulearn`). Loss went from 4.25 to 1.79. Remaining-fact accuracy at t=0 was 0.32 at n=4 (chance 0.25) and 0.26 at n=5 (chance 0.20).
  - This only shows that the pipeline learns something. It is far too small to say anything about the task.
- **Default model on CPU.** The default model (3.17M params, mixed environment, orders 4–8 and 10) ran 4 full steps on CPU at about 10k tokens/s. This is a size and shape check only.

## GPU throughput

Measured on 2026-10-07 with GPU_BUSY absent. Each test was capped below 3 min, and the script re-checked the flag before each one. All tests used the mixed environment, orders 4–8 and 10, d 256, 8 heads, and 65,536 tokens/step in micro-batches of 16,384. Logs are in `runs/_gpubench`, `runs/_gpubL1` and `runs/_gpubL6`.

| model | params | tokens/s (mean, steps 20–end) | step time | peak VRAM (allocated) | 14k steps | 20k steps |
|---|---|---|---|---|---|---|
| 1 layer | 0.81M | 710k | 0.092 s | 0.24 GiB | 21 min | 31 min |
| **4 layers (default)** | 3.17M | **243k** | **0.268 s** | **0.75 GiB** | **63 min** | **89 min** |
| 6 layers | 4.74M | 146k | 0.451 s | 1.09 GiB | 105 min | 150 min |
| 2 layers | — | not measured | ≈ 0.15 s (interpolated) | — | ≈ 35 min | ≈ 50 min |

- **Test lengths.** 150 steps for L4 (12:14–12:16), and 120 steps each for L1 and L6 (13:18 and 13:22).
- **Startup.** Each run spends about 20 s at startup, and the mixed environment's pools take up to about 14 s more. Each in-training evaluation takes a few seconds.
- **VRAM.** VRAM is far below 8 GB. The CUDA context adds a few hundred MB on top of the allocated figure. `--micro_tokens 65536` (no accumulation) would also fit, but it was not timed.
- **`eval_fp.py` on GPU.** 450 trials (25 per file, 14,300 queries) took 11 s with the 4-layer model. All 1,800 trials therefore take about 45 s per symbol draw, or about 3 min with `--draws 4`. On CPU the same model needs 2.7 s per trial.
- **Losses.** Training losses fell from about 4.3 to 3.3–3.6 within 120–150 steps. This only shows the GPU path works; no task-level result is claimed.
- **CPU.** The default config runs at about 10k tokens/s on CPU (4 steps measured), about 25× slower than the GPU.

## Proposed minimal run grid (about 6–8 GPU-hours)

**Shared settings:**
- Model: d_model 256, 8 heads, RoPE.
- Steps: **14,000 steps** of 65,536 tokens (`--steps 14000`), about 0.92B tokens. Warmup 500, lr 6e-4 (1e-3 for L ≤ 2), cosine to 10%.
- `--orders 4 5 6 7 8 10`, max_len 2400. `--minutes` is set to about 1.2× the expected time as a safety cap.

**After each run:**
- `eval_fp.py --draws 4`, about 3 min per model on GPU (scaled from a 450-trial timing).
- Then `analyze3.py`.

**How to read the predictions:**
- Idealised values are G-cell accuracy at n=6, k=10 for t = 0 → 1 → 4 (c10 §5.4):
  - N_iso: .30 → .65 → .96, held-out .31;
  - A: .30 → 1 → 1, held-out 1;
  - B / Bh: 1 flat.
- The decision rule is Exp-1's, n=6 primary and n=8 secondary.
- "w" is held/comp. "shape" is g(G4) − g(G1): about 0 for A, about 0.3 for N_iso.

| # | name | config differences | answers | est. time | what the theory predicts |
|---|---|---|---|---|---|
| 1 | `A4` | `--layers 4 --types noheld --env_mix group=0.5,latin=0.25,loop=0.25` | (a) positive control | 65 min (measured rate) | **A**: G(t0) about loop/Latin level (env unknown) → step at t=1 → held-out S3/D4/Q8 ≈ seen gain (w ≈ 1, shape ≈ 0); scramble_t4 ≤ t0 level. Alternative within the theory: if the net does Bayes over environments with enough cue evidence (Bh), it is **B-type** (flat ≈ 1), because 10 cues already identify the environment (Prop D: ≤ 1 cell of k* shift). A vs B-type is the bounded-vs-unbounded test. |
| 2 | `G4` | `--layers 4 --types noheld --env_mix group=1` | Prop-C control for (a) | 65 min | **No compounding**: B-type for seen types (Z6 G-cells ≈ 1 at t=0) or N; comp < .15 at any t. Held-out non-abelian targets near Latin/loop level, flat in t. If this run compounds, the "history is uninformative ⇒ no compounding" premise is wrong for SGD-trained nets (e.g. a pure context-length effect, which `scr` then shows). |
| 3 | `P4` | `--layers 4 --types noheld --env_mix group=1 --pool_M 32` (32 labelled symbol tables per order) | (b) exemplar control: low task diversity | 65 min | Memorises the pool (training loss → ~0 on pool sequences) but cannot transfer to fresh symbols: in-training `in*` acc near chance, Exp-1 class **R** (acc_L < 1/n + .25) or **N**; no compounding and no held-out transfer. If the in-training `in6` accuracy rises anyway, the pool is too large (the net generalised); reduce to M = 8. |
| 4 | `I4` | `--layers 4 --env_mix isocopy=0.5,latin=0.5` | (b) N_iso positive control | 65 min | Never sees a group or an identity, so it can only learn in-context isomorphism matching with Latin fallback. **N_iso**: gradual rise following the missing-mass curve (.30 → ≈.65 → ≈.96 at n=6), held-out S3 with Z6-only history ≈ .31 (w ≈ 0), shape ≈ .3, leak_t1 high. If `isocp6_t1` in `eval.csv` stays near chance, the net did not learn matching and the run is **N/R** (uninformative for N_iso). |
| 5 | `A1` | as #1, `--layers 1 --lr 1e-3` | (c) depth | 22 min (measured rate) | **R or N**: one layer can copy and do some Latin elimination, but not chain facts; no compounding. |
| 6 | `A2` | as #1, `--layers 2 --lr 1e-3` | (c) depth | ≈ 37 min (interpolated) | **N-like with O ≫ G**: identity and cancellation (the In-Context-Algebra mechanisms, O-cells) emerge; G-cells stay near loop level (≈ .42); little G compounding. |
| 7 | `A6` | as #1, `--layers 6` | (c) depth | 107 min (measured rate) | Same or higher held-out transfer (w) than A4. Bounded-inference prediction: if 6 layers suffice to infer the class from the 10-cue target alone, t=0 G rises toward 1 and the run becomes **B-type**, so comp *shrinks*. |
| (8) | `C4` (optional) | as #1 with `--types cyclic` | (b) "narrow abstraction" control | 65 min | Learns "group env = cyclic". Compounds on Z6 targets, but held-out and S3 cells fail, so Exp-1's rule says **N_iso** although the mechanism is an over-narrow abstraction, not exemplar lookup. This calibrates how far w ≈ 0 can be read as "exemplar". |

Commands for the grid (from `exp3/`, with the GPU env python; check GPU_BUSY first):

```
C="--orders 4 5 6 7 8 10 --steps 14000 --minutes 110"
M="--env_mix group=0.5,latin=0.25,loop=0.25"
python train.py --name A4 --layers 4 --types noheld $M $C
python train.py --name G4 --layers 4 --types noheld --env_mix group=1 $C
python train.py --name P4 --layers 4 --types noheld --env_mix group=1 --pool_M 32 $C
python train.py --name I4 --layers 4 --env_mix isocopy=0.5,latin=0.5 $C
python train.py --name A1 --layers 1 --lr 1e-3 --types noheld $M $C
python train.py --name A2 --layers 2 --lr 1e-3 --types noheld $M $C
python train.py --name A6 --layers 6 --types noheld $M $C --minutes 150
python train.py --name C4 --layers 4 --types cyclic $M $C          # optional
for r in A4 G4 P4 I4 A1 A2 A6; do python eval_fp.py --ckpt runs/$r/final.pt --name $r --draws 4; done
python analyze3.py
```

`--minutes` is only a safety cap. A run stopped by the cap has not finished its cosine schedule; check the `DONE ... step` line. `--resume` continues a run from `last.pt`.

**Depth prediction (runs 5, 6, 1, 7 = L1, L2, L4, L6).** This is the theory's sharp, falsifiable bounded-inference prediction (c10 §5.1). Compounding is large only when t=0 inference is bounded: the cue alone is not enough, but history-narrowed hypotheses are.
- The compounding magnitude `comp` should be **non-monotone in depth**: about 0 at L=1, small at L=2, largest at an intermediate depth, and shrinking again once depth makes t=0 inference sufficient (B-type).
- The abstraction fraction w should be non-decreasing with depth.
- A monotone rise of comp with depth up to L=6 does not by itself refute the theory, because the turnover may lie beyond 6 layers. A *fall* of w with depth would count against it.

**Budget.**
- Runs 1–7 take 4 × 65 + 22 + 37 + 107 ≈ **426 min, about 7.1 GPU-h**. Adding 7 × 3 min of `eval_fp` brings it to about **7.5 h**.
- Run 8 adds 65 min, which takes the total over 8 h. Run it only if time allows, or in place of run 7.
- All durations except L2's come from measured throughput. Each includes about 2 min for evaluation, checkpoints and startup.
- With 20k steps, runs 1–7 would take about 10.5 h.
- If the GPU is shared during a run, everything slows down. `--minutes 110` (safety cap) or `--minutes 150` for A6 stops a run cleanly, and `--resume` continues it.

**Order.** Run 1 first. If its `eval.csv` shows in-distribution `in6_t4` accuracy still near chance at the end, the model or schedule is undertrained for this task. Fix that (more steps, or lr) before spending the rest of the budget.

## Caveats and deviations

- **Environment mixing and `isocopy` go beyond the initial task specification.** I added them because of Prop C and the abelian-only fact above. The runs named in that specification (group-only `noheld`, pool, cyclic-only) are all supported and included in the grid.
- **Pool mode can reuse symbols across the tables of a sequence.** See `--pool_M` above.
- **The `latin` environment draws are not uniform.** They are random isotopes of 256 randomly backtracked squares per order, rejected if they have an identity. The `loop` environment draws relabellings of up to 256 random non-associative loops per order (fewer at n=5, where fewer exist).
- **Training evaluation is teacher-forced.** It scores sequences that hold the whole target, so later facts see more cues. The exact fingerprint is `eval_fp.py`, which scores each cell independently at the Exp-1 cue.
- **Windows.** No `torch.compile` or flex-attention is used (Triton is generally unavailable on Windows), so attention is plain SDPA. Nothing was pip-installed.

## Run history and outcome (added 2026-10-09)
- **Interruptions.** A4 was resumed from step 2,000. A2 was retrained from scratch after stalling at step 1,759. G4 (stopped at 9,934) and A6 (stopped at 13,471) were resumed to 14,000. Their partial-checkpoint scores are kept as `res/_G4_step9934.jsonl` and `res/_A6_step13471.jsonl`; the A6 files were renamed from `13450`, which was the last *logged* step. Details are in `paper/methods.md`, "Interruptions" and "Deviations".
- **Reading `eval.csv`.**
  - `runs/A4/eval.csv` has two step-3,000 blocks, from the abandoned attempt and from the rollback. Keep the last occurrence per (step, cond).
  - P4's `isocp6_t1` (1.000) is invalid: in pool mode the generator returns memorised pool tables.
  - On resume, `config.json` holds the resume invocation's arguments.
- **Outcome.** Every network is class R. Each learned network (A6 partly) answers with the symbol least represented among the revealed facts, a global count shortcut; P4 (not learned) does not; see `heur/count_heuristic.md` and `heur/intrain_heuristic.py`. The shortcut alone scores 0.33 in-training at n=6. That matches the plateau of A4, G4 and I4 to within ±0.006; A1 and A2 sit below it, and A6 slightly above it (+0.015). The preregistered learning-failure threshold (0.267) therefore cannot tell shortcut from task. P4 is *not learned* under that rule. Results are in `paper/results.md`, R4.
