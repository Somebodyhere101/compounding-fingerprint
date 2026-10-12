# Experiment 5: does the compute–retrieve (CR) axis track mode under perfect format control?

Frozen 2026-10-08, before any Experiment 5 output. The user approved the experiment in chat.

## Motivation
The CR-axis paper (Ramdan 2026; `Documents/Codex/2026-10-05/task/upstream`) finds a mean-difference direction separating computation prompts from retrieval prompts. Its own stated limitation is that compute prompts are longer and more complex than retrieval prompts. The fingerprint stimuli remove that confound:
- In **n6_leak_t1**, every queried cell is *retrievable*: the history table is the target's own labelled table.
- In **n6_base_t1**, the same format, length and query must be answered by *inference*: the history is a different group table.

Behaviour is known from Experiments 1–2. At 4–8B, retrieval is near ceiling (leak 0.95–0.96) and inference is at chance.

## Method
- **Axis.** The CR detector exactly as published (`cr_axis/detector.py`): hidden state at the last token of decoder-layer output; prompts wrapped by `_format_prompt` (chat template when the tokenizer has one); calibration on the paper's 30 + 30 prompts (`data/calibration/{compute,retrieve}.txt`); axis = normalised mean difference; layer = best AUC among the detector's four candidate layers (n/4, n/2, 3n/4, n−2), first maximum wins.
- **One implementation change, checked.** The forward pass calls the decoder stack (`model.model`) and skips the output layer, which hooks on decoder-layer outputs do not depend on. Numerical equivalence to `CRDetector._get_hidden` is checked on Qwen3-0.6B-Base and reported.
- **Test inputs.** Each query = trial context + `\nwa * wb =`, in two formats:
  - (a) **detector format**, wrapped by `_format_prompt` (primary);
  - (b) **raw format**, as scored behaviourally (secondary).
- **Speed.** A shared-prefix KV cache is used per trial. Hidden states at the query's last token are recorded at all four candidate layers.
- **Models.**

  | model | precision |
  |---|---|
  | Qwen3-0.6B-Base | bf16 |
  | Qwen3-1.7B-Base | bf16 |
  | Qwen3-1.7B (post-trained) | bf16 |
  | Qwen3-4B-Base | nf4 |
  | Qwen3-8B-Base | nf4, embeddings and head on CPU |

  The nf4 caveat from Experiment 2 applies.
- **Queries.** All 26 per trial for models ≤1.7B. For 4B and 8B, 12 per trial, stratified by forcing class with a fixed seed. 100 trials per condition.

## Hypotheses and tests (direction stated in advance; compute = positive)
- **H1 (primary).** Mean CR projection at the detector's layer, detector format, is higher in n6_base_t1 than in n6_leak_t1.
  - Statistic: the difference of trial-mean projections, with a trial-bootstrap 95% CI (2,000 resamples).
  - Standardised effect: difference ÷ pooled SD of trial means.
  - Query-level AUC (base vs leak).
  - Decision: supported if the CI excludes 0 in the predicted direction for ≥3 of the 5 models.
- **H2.** Within n6_base_t1, projection increases with inference demand in the class order L < O < G.
  - Statistic: Spearman ρ between class rank and projection over queries, with a trial-bootstrap CI.
- **Calibration replication.** We report the calibration AUC and chosen layer per model. This replicates the CR axis on the Qwen3 family, which the paper did not test.

## Exploratory (labelled as such)
- Raw-format projections.
- Profiles across all four layers.
- Projection vs. correctness within each condition, joined from the model's Experiment-1/2 results file.
- n6_base_t0 vs n6_base_t4 (does history change mode?).

## Interpretation, fixed in advance
- **H1 supported:** the axis tracks processing mode under matched format. That supports the CR paper beyond its length/complexity confound. Combined with chance-level accuracy, it also says the models *enter compute mode and fail*, rather than retrieving.
- **H1 not supported, with calibration AUC ≈ 1:** the axis separates the calibration prompt types but does not track mode once format is matched. That is a validity limitation the CR paper must report.

## Log
- 2026-10-08 ~18:30. A pipeline smoke test was run on CPU with Qwen3-0.6B-Base (6 trials per file, 6 queries per trial). It validated the code and the analysis script. Its outputs were deleted and are not used. The hypotheses, tests and decision rule above were *not* changed after it, although its tiny-sample H1 estimate pointed the other way (base < leak). The equivalence check against `CRDetector._get_hidden` gave a max relative difference of 0.0.
- 2026-10-09, timestamp correction (added after the final audit; the entry above is left as written).
  - This file was last saved at 18:20:09, and `cr_probe.py` and `analyze5.py` at 18:14–18:16. The smoke test therefore ran at about 18:17, and its outputs were seen minutes before the final save, not at "~18:30" as the entry says.
  - The hypotheses kept their original direction, even though the smoke test pointed the other way.
  - The "frozen before any Experiment 5 output" claim holds only up to these few minutes.
- 2026-10-09. Precision: the detector code is used exactly as published, but in bf16 rather than the paper's fp16.
- 2026-10-09. Query count: for 4B and 8B, stratified rounding gives 11 rather than 12 queries in 20 of 400 trials (4,780 queries).
- 2026-10-09. Outcome:
  - H1 is supported in 0 of 5 models.
  - The difference is significantly reversed (base < leak) in 0.6B-Base, 1.7B-Base and 4B-Base. The interpretations above do not cover this case, and it is reported as falling outside them.
  - H2 is significantly reversed in 1.7B-Base.
  - A vocabulary-overlap control was added afterwards as exploratory Experiment 5b (`CONTROL_vocab.md`).
- 2026-10-09, correction to "Calibration replication" above. The CR paper did test a Qwen3 model (Qwen3-30B-A3B, AUC 0.997), so these runs extend the axis to small Qwen3 models; they are not a new family. Three of the five calibration AUCs here (0.964–0.978) are slightly below the paper's reported floor of ≥0.988.
