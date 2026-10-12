# Experiment 6 results: the fingerprint on the released In-Context Algebra model

Run on 2026-10-11 under `PREREGISTRATION.md`. Experiment-6 numbers come from these generated files:
- `RESULTS_tables.md` (`analyze6.py`);
- `mechanism.md` (`mechanism.py`, using `res/loop_level.jsonl` from `loop_level.py`);
- `validate_*.json`.

The parameter and token figures come from `exp3/runs/A4/config.json` and `dl/metadata.json`. An adversarial verification pass on 2026-10-11 re-derived these numbers and checked the adapter. Its corrections are included.

## Bottom line
The fingerprint returns a **non-R class** on a real network for the first time. This model infers Latin- and loop-forced cells from sparse cues almost perfectly.

It is **not B-type**. Associativity-only (G) cells reach 0.53–0.71, against 0.87–1.00 on Latin- and loop-forced cells, and O − G is +0.27 to +0.45 in every cell. The group structure in its weights is therefore partial, and associativity is its weakest route, consistent with the authors' report.

The compounding branch (N vs N_iso vs A) cannot be tested on this model, because its 16 symbol tokens allow at most one earlier table.

## Adapter validation (the model's own distribution)
**Re-implementation.** The model was re-implemented from the upstream code without executing it, and loaded with `weights_only=True`. A second, independent forward pass (explicit causal attention, real-valued RoPE, erf-GELU) agrees with the adapter on 3,200 of 3,200 answer positions, with max |logit diff| 1.9e-4.

**Held-out final-fact accuracy at k=200 facts:**

| setting | accuracy |
|---|---|
| shuffled vocabulary | 0.973 (n=512) |
| upstream evaluation protocol (generator, unshuffled vocabulary, final-token argmax), on 1,024 fresh sequences | 0.959 |
| shuffled vocabulary, k=100 / 50 / 20 | 0.91 / 0.83 / 0.66 |

**The 0.959 is below the 99.4% the authors report** for this configuration. Their figure is the maximum over up to about 290 training-time evaluations on one cached 512-sequence set, and `algebra_gpt_best.pt` was saved at that maximum. On the same 1,024 fresh sequences the released checkpoints score:

| checkpoint | accuracy |
|---|---|
| best | 0.959 |
| 100k | 0.960 |
| 40k | 0.975 |
| 20k | 0.971 |
| 10k | 0.835 |

The shortfall is therefore consistent with snapshot selection on a fixed set, not a port error. Full-sequence training loss is 0.917, against about 0.93 at the end of the upstream curve.

## Primary: decision-rule steps 1–3 (base t=0; stimulus order; 4 symbol draws; bootstrap seeded as in exp1/analyze.py)

| cells | acc_L [95% CI] | R threshold | G t0 [CI] | g(G t0) [CI] | class |
|---|---|---|---|---|---|
| n=6, k=10 | 0.886 [0.820, 0.944] | 0.417 | 0.637 [0.601, 0.675] | 0.65 [0.59, 0.74] | non-R, non-B |
| n=8, k=16 | 0.971 [0.909, 1.000] | 0.375 | 0.663 [0.631, 0.699] | 0.64 [0.58, 0.71] | non-R, non-B |
| n=5, k=6 | 0.982 [0.952, 1.000] | 0.450 | 0.531 [0.485, 0.574] | 0.42 [0.36, 0.48] | non-R, non-B |
| n=6, k=16 (addendum-2d stimuli) | 0.920 [0.900, 0.938] | 0.417 | 0.704 [0.651, 0.758] | 0.71 [0.64, 0.79] | non-R, non-B |

- The class is the same on mean p_gold.
- Loop-forced cells score 0.90 (n=5), 0.91 (n=6) and 0.99 (n=8).
- The k=16 t=1 cells give the same profile: L/O/G 0.930/0.920/0.710 (base) and 0.909/0.917/0.713 (heldout).

## Preregistered predictions

| prediction | outcome | evidence |
|---|---|---|
| P1: not R at n=6 | **met** | acc_L 0.886, against a threshold of 0.417 |
| P2: B-type at n=6 | **not met** | g(G t0) 0.65 < 0.85: non-R, non-B |
| P3: leak recall ≥ 0.90 | **met** | 1.000 on every forcing class |
| P4: G above learner C | **met** | +0.411 [+0.369, +0.453] (paired) |
| P5: no benefit from a differently labelled history (n=6 G, t1 − t0) | **met** | −0.025 [−0.071, +0.024] |
| P6: unseen order-8 types below seen types | **not met** | +0.025 [−0.035, +0.085] at n8 base t0; pooled n8 cells −0.005 [−0.036, +0.026] |

On P6, G accuracy depends on the specific group more than on whether the type was seen in training. At n8 base t0:
- Z2³ (unseen): 0.97. Its smaller relative Z2×Z2 (= D2) was in the training mix.
- Z4×Z2 (unseen): 0.67.
- D4 (seen): 0.67.
- Z8 (seen): 0.63.
- Q8 (unseen): 0.51.

Latin- and loop-forced accuracy is 0.95–1.00 for seen and unseen types alike.

## Secondary
- **Against the baselines** (n=6 base t0, expected accuracy):

  | cells | model | learner C | one-pass row/column elimination |
  |---|---|---|---|
  | L | 0.886 | 0.148 | 0.563 |
  | O | 0.910 | 0.212 | 0.314 |
  | G | 0.637 | 0.226 | 0.289 |

  The model beats both baselines by a wide margin in every cell.
- **Cue sweep** (n=6, t=0, all queried cells), next to Qwen3-8B-Base on the same trials (Fig. 6b):

  | k | 6 | 10 | 16 | 22 | 28 |
  |---|---|---|---|---|---|
  | In-Context Algebra model | 0.69 | 0.74 | 0.88 | 0.96 | 0.998 |
  | Qwen3-8B-Base | 0.20 | 0.21 | 0.25 | 0.28 | 0.29 |

  Row-sorting the facts makes no difference for this model (k=16: 0.881 vs 0.878).

## Exploratory (preregistered as exploratory)
- **Interleaved fact order.** It changes nothing: at n=6, L 0.886, O 0.908, G 0.647, and the class is the same at n=5, 6 and 8.
- **Training trajectory** (6 of the 21 released numbered checkpoints; n6 base t0, 2 draws):

  | step | acc_L | O | g(G t0) | class | sweep k=28 |
  |---|---|---|---|---|---|
  | 0 | 0.167 | 0.155 | — | R | 0.17 |
  | 5k | 0.351 | 0.392 | 0.45 | R | 0.58 |
  | 10k | 0.807 | 0.840 | 0.49 | non-R, non-B | 0.93 |
  | 20k | 0.860 | 0.885 | 0.55 | non-R, non-B | 0.99 |
  | 40k | 0.868 | 0.896 | 0.62 | non-R, non-B | 0.99 |
  | 100k | 0.886 | 0.911 | 0.65 | non-R, non-B | 0.998 |

  Of the six checkpoints scored, those at 0 and 5k are R and those from 10k on are non-R. The upstream training loss drops sharply between those steps. After 20k, Latin and loop accuracy plateau at 0.86–0.91 while g(G t0) keeps rising. A graded index such as g(G t0) captures this; the categorical classes do not.

## Post hoc (added after the primary results): the mechanism split (`mechanism.md`)
**Commutative copying.** In non-abelian targets, when the transposed cell b·a is revealed but differs from a·b, the model answers with the transpose 38% (n=6) or 46% (n=8) of the time.

**With the transpose hidden**, G accuracy is compared with three learners on the same cells: the Latin-only learner N, a loop-level learner, and C.
- The **loop-level learner** picks uniformly among the values a loop completion of the cue allows. It is computed exactly with the stimulus generator's SAT engine.
- G cells are by definition not loop-forced, so beating this learner means using group structure beyond identity and cancellation.

| order | targets | model | learner N | loop level | learner C |
|---|---|---|---|---|---|
| n=5 | abelian | 0.46 | 0.32 | 0.46 | 0.29 |
| n=6 | abelian | 0.63 | 0.30 | 0.43 | 0.27 |
| n=6 | non-abelian | 0.54 | 0.30 | 0.42 | 0.24 |
| n=8 | abelian | 0.61 | 0.22 | 0.29 | 0.19 |
| n=8 | non-abelian | 0.50 | 0.22 | 0.29 | 0.17 |

Model minus loop level (trial bootstrap):
- n=6: +0.148 [+0.130, +0.166];
- n=8: +0.254 [+0.236, +0.273];
- n=5: +0.004 [−0.043, +0.048].

At n=6 and n=8, then, part of the model's associativity-forced accuracy goes beyond identity and cancellation. At n=5 it does not. It also uses a commutativity shortcut that fails on non-abelian groups.

## What this does and does not show
- **It shows** that the fingerprint's first two steps (R vs not R, B-type vs not) separate a real, task-trained network from every pretrained LM and every small network we trained. It also gives that network a profile consistent with its authors' mechanistic analysis: strong identity, cancellation and copying, weaker associativity.
- **It does not show** the compounding branch. That needs t ≥ 2 and scramble histories, so it needs a model with more symbol tokens. That branch remains the open positive control.
- **It does not isolate why this model succeeds where our networks found a counting shortcut.** It has about 16× the parameters of our 4-layer networks. But its checkpoints leave class R between 5k and 10k steps, after about 0.64–1.28B training tokens (5k–10k × 128 sequences × about 1,000 tokens). That is comparable to our 0.92B, so training tokens do not explain the difference.

  The recipes also differ in width (1024 vs 256), learning rate, vocabulary (16 vs 70 tokens), loss target (every token vs answers only) and context density (groups-only 200-fact sequences vs sparse cues). This comparison cannot separate those factors.
