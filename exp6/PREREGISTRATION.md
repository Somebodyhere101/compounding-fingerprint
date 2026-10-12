# Experiment 6: the compounding fingerprint on the released In-Context Algebra model (positive control)

Frozen 2026-10-11, before any of our stimuli were scored with this model. The user asked for this run in chat.

**What had been run before freezing.** Only the validation checks on the model's own training distribution:
- `validate.py`;
- `validate_unshuffled.py`;
- `validate_loss.py`.

## Why
Every real system scored so far is class R: nine pretrained LMs and seven small from-scratch networks. The decision rule has been validated only on exact idealised learners.

In-Context Algebra (Todd, Brinkmann, Gandikota, Bau; ICLR 2026; arXiv 2512.16902) released a transformer trained to complete in-context group tables. It is a 4-layer, 8-head, d=1024 RoPE GPT with 16 symbol tokens, checkpoint `algebra_gpt_best.pt` (sha256 9c0ee347…a265d9). It is the closest prior work, and the best available real network that should *not* be R. This experiment asks whether the fingerprint returns a non-R class on it, and which one.

## Model and adapter
- **Weights and code.** The checkpoint comes from https://algebra.baulab.info/weights/mixrosette-facts-10-16-8heads/models/. The code is GitHub ericwtodd/algebra at commit 532a6a2, read but not executed. The model is re-implemented in `ica.py` and loaded with `torch.load(weights_only=True)`.
- **Validation** (own distribution; 200 facts, final fact held out with its commutative partner):
  - shuffled vocabulary: 0.973 (n=512);
  - the upstream eval protocol exactly (unshuffled vocabulary): 0.959 (n=1,024);
  - full-sequence training loss: 0.917, against about 0.93 at the end of the upstream loss curve. The re-implementation is therefore faithful.
  - Accuracy falls with fewer facts: 0.91 at k=100, 0.83 at k=50, 0.66 at k=20.
- **Token format.** Symbols are 0–9 and a–f (ids 0–15), '=' is 16 and ',' is 18. A fact is `,ab=c`. A sequence starts with ',' and has no header or start token.
- **Mapping our stimuli** (`score.py`).
  - Every distinct word of a trial gets a distinct random symbol, from a per-(file, trial, draw) seed.
  - The assignment is redrawn until no identity element of any group table in the trial maps to '0'. This matches the model's training, where '0' is never an identity (`holdout_zero`).
  - Identities are computed from each complete history table, and from the target table reconstructed as cue plus query golds.
  - Facts are emitted in the stimulus order: history tables (each fact once), then the target's cue. 'Table X:' headers are dropped, because tables are distinguished by disjoint symbols as in training.
  - Each query `wa * wb =` becomes `,ab=` and is scored independently. The score is forced choice over the target's n candidate symbols, by the next-token logits at the final '=' position.
  - There are four symbol draws per trial. Candidate probabilities are averaged over draws, and pred is the argmax of the average, exactly as in `exp3/eval_fp.py`.
  - Records use the Experiment-1 results format.

## Cells (only cells with ≤16 distinct words fit the 16 symbols)
- **Experiment-1 stimuli.** n5_base_t0; n6_base_t0, n6_base_t1, n6_leak_t1, n6_heldout_t1; n8_base_t0, n8_base_t1, n8_heldout_t1.
- **Cue sweep.** `exp2/stim` ks_n6_k{6,10,16,22,28} and kso_n6_k{16,22,28}, t=0 trials only.
- **Addendum-2d stimuli.** `exp2/stim_k16` n6_base_t0, n6_base_t1, n6_heldout_t1.
- **Not measurable.** Every t ≥ 2 cell (18–40 words), every scramble cell, the loops cells, and the instructed variant (it has a natural-language header). The compounding contrast (t4 − scramble), comp, held and w therefore cannot be computed for this model.

## Which of our target types this model saw in training
Its training groups are C3–C10 and D2–D5.

| our cells | target types | seen in training? |
|---|---|---|
| n5 | Z5 = C5 | seen |
| n6 | Z6 = C6, S3 = D3 | both seen |
| n8 base | Z8 = C8, D4 | seen |
| n8 base | Z4×Z2, Z2³, Q8 | unseen (held out by the authors) |
| n8 heldout | D4 | seen |
| n8 heldout | Q8 | unseen |

n8 results are reported separately for seen and unseen target types.

## Primary analysis: decision-rule steps 1–3 (they need only the base t=0 cell)
The rule is that of `exp1/analyze.py`, unchanged, at n=6 (primary) and n=8 (secondary), on accuracy and on p_gold:
1. acc_L = L-cell accuracy in base t0. If acc_L < 1/n + 0.25, the class is **R**.
2. g(G_t0) = (G_t0 − 1/n)/(acc_L − 1/n). If ≥ 0.85, the class is **B-type**.
3. Otherwise the class is "**non-R, non-B**". N, N_iso and A are undetermined, because their comp step needs t=4 and scramble cells.

CIs are trial-bootstrap (2,000 resamples, data-seeded as in analyze.py).

## Predictions (directional, stated before scoring)
- **P1.** Not R at n=6: acc_L ≥ 0.417.
- **P2.** B-type at n=6, because the training diet is groups only, so the group prior lives in the weights. The alternative outcome, non-R non-B, would mean associativity-forced cells lag Latin-forced ones (the paper reports associativity as its weakest route).
- **P3.** Recall: leak t1 G accuracy ≥ 0.90. Copying is the paper's strongest mechanism.
- **P4.** G accuracy at n6 t0 exceeds learner C (0.226; `exp1/ideal/res/ideal_C.jsonl`) with a trial-bootstrap CI excluding 0.
- **P5.** No benefit from a differently labelled history: n6 base t1 − t0 on G accuracy has a CI including 0 (B-type signature; the history adds nothing the weights lack).
- **P6.** n8, unseen target types (Z4×Z2, Z2³, Q8) below seen (Z8, D4) on G accuracy.

**Exact tests**, all trial-bootstrap with 2,000 resamples and fixed per-test seeds:
- **P4** is paired by trial: model G accuracy minus learner C's (`correct` field) on n6_base_t0.
- **P5** uses independent trials: G accuracy in n6_base_t1 minus n6_base_t0.
- **P6** uses n8_base_t0 (primary) and independent trial sets: G accuracy pooled over unseen-type targets minus seen-type targets. The same contrast pooled over n8_base_t0, n8_base_t1 and n8_heldout_t1 is secondary.

The primary format is the stimulus order with 4 draws.

## Secondary and exploratory (labelled as such)
- Accuracy by forcing class (L, O, G, Ab, Cy) in every cell, and O − G (identity/cancellation vs associativity).
- The cue-sweep curve vs learner C and one-pass row/column elimination.
- **Interleaved format (exploratory).** The same trials with all context facts (history and cue) in one random order, closer to the training distribution of mixed groups.
- **Developmental trajectory (exploratory).** Released checkpoints at steps 0, 5k, 10k, 20k, 40k and 100k, scored on n6_base_t0 and the ks t0 sweep with 2 draws, to give the class per checkpoint.

## Interpretation, fixed in advance
- **B-type or non-R at n=6.** The fingerprint returns a non-R class on a real network trained for the task. This is the positive control for steps 1–3 of the rule. The compounding branch (N vs N_iso vs A) remains untested on a real network.
- **R at n=6.** Even this model cannot infer from 10 sparse cues of a 6×6 table. Its published competence would then depend on dense contexts, which the cue sweep tests directly. That would be reported as a null for the positive control.

## Log (appended after scoring; the text above is unchanged)
- **2026-10-11, after a verification pass.** "The upstream eval protocol exactly" above means the same generator and final-token argmax over the full vocabulary. The sequences, however, were 1,024 fresh ones, not the authors' cached 512-sequence set, and they used the re-implementation's own element order.
  - The ≥0.99 expected in `validate_unshuffled.py` was not met (0.959).
  - `validate_ckpts.py` shows that the released checkpoints score 0.959–0.975 on the same fresh sequences, and that a second, independent forward pass agrees with the adapter on 3,200 of 3,200 answer positions.
  - The authors' 99.4% is the maximum over about 290 evaluations on one fixed set. The shortfall is therefore consistent with checkpoint selection, not with a port error.
- **2026-10-11.** The first analysis run seeded the steps 1–3 bootstrap from labels and trial count rather than "as in analyze.py". This was corrected; CI endpoints moved by ≤0.01.
