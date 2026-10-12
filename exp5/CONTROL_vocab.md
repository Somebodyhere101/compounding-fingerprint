# Experiment 5b: vocabulary-overlap control for the CR-axis result (exploratory, NOT preregistered)

Written 2026-10-09 before this full-size run.

## What we already knew when this was designed
- H1 had come out reversed (base < leak) in Qwen3-0.6B-Base, Qwen3-1.7B-Base and Qwen3-4B-Base.
- The final audit had run a 20-trial pilot of this design (`audit2/e5_ctrl.py`) on 0.6B-Base and 1.7B-Base. In it, relabelling closed 53% and 71% of the gap.

This run is therefore a full-sample estimate of an effect already seen in a pilot. It is not an independent test.

## Why
Leak and base contexts are matched in token count, query position and table sizes. They differ in vocabulary:
- In **leak**, history Table A uses the target's own 6 words, so the context has 6 distinct words. Every target line, and the query pair with its answer, appears verbatim in Table A.
- In **base**, Table A uses 6 other words, so the context has 12 distinct words.

So base vs leak changes both *retrievability* and *surface repetition*.

## Conditions
There are 100 trials per stimulus file, using all 26 queries, unchanged queries, and unchanged Table B (the target).

| condition | source | Table A | retrievable? | target vocabulary in A? |
|---|---|---|---|---|
| leak | n6_leak_t1 | the target's own labelled table | yes | yes |
| leakRel | n6_leak_t1 | the same table, words permuted by a random derangement | no (answers inconsistent) | yes |
| baseRel | n6_base_t1 | a different group table, its 6 words mapped onto the target's words by a random bijection | no (about 15% of lines agree by chance) | yes |
| base | n6_base_t1 | a different group table in 6 other words | no | no |

Relabelling uses a per-trial seed (`default_rng([11, trial])`).

## Measurement
- The same axis, layer and format as H1: the published detector's axis from `exp5/res/{model}_axes.npz`, the detector layer, `_format_prompt` (det format), and the last-token decoder-layer output.
- Forward passes are uncached. Qwen3-0.6B-Base ran one query per forward (`ctrl_vocab_unbatched.py`).
- *Implementation note added after the run:* Qwen3-1.7B-Base, Qwen3-1.7B and Qwen3-4B-Base ran the 26 queries of a trial-condition in one batched forward (`ctrl_vocab.py`, batch 26). All rows have equal token length, so no padding occurs, and a 3-trial check on Qwen3-0.6B-Base (`res_ctrl_chk/q06b_b2.jsonl`) reproduced the per-query projections exactly.
- Raw format and the other three layers are recorded as secondary.

## Statistics
Trial-mean projections are computed per condition.
- f_vocab = (baseRel − base) / (leak − base) is the fraction of the H1 gap reproduced by giving a non-retrievable history the target's vocabulary.
- r_retr = leak − baseRel is the part that remains attributable to retrievability, together with leak − leakRel.
- Bootstrap CIs (2,000 resamples) resample base-file trials (base and baseRel jointly) and leak-file trials (leak and leakRel jointly) independently.

## Reading
- If f_vocab is near 1 and r_retr is near 0, the base/leak difference is a vocabulary or repetition effect, not a mode effect.
- If f_vocab is near 0, it is not.

Models: Qwen3-0.6B-Base, Qwen3-1.7B-Base, Qwen3-1.7B (bf16) and Qwen3-4B-Base (nf4). 8B is omitted because it is slow with offloading and its det-format H1 was null.
