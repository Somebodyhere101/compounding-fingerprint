# What Experiments 5 and 5b mean for the compute–retrieve (CR) axis preprint

*Notes on the CR-axis preprint (Ramdan 2026, doi:10.5281/zenodo.19455724), whose author is also the author of this study. Every Experiment 5 and 5b result comes from one of two sources; other figures are quoted from the preprint or from `paper/methods.md`:*
- *`exp5/RESULTS_tables.md`: Experiment 5, preregistered;*
- *`exp5/RESULTS_ctrl.md`: Experiment 5b, an exploratory vocabulary control designed after Experiment 5's result and a 20-trial audit pilot.*

## 1. The test
In its adversarial-prompts control, the paper notes that "the retrieval prompts are mostly short factoid lookups while the computation prompts are longer word problems, so prompt complexity remains a potential confound". The fingerprint stimuli remove length and format differences exactly. Two conditions share the same prompt format, the same token count (281 context tokens in both) and the same query position:
- **leak_t1**: the answer is in the context. An earlier table is the target's own labelled table, so the queried line appears verbatim.
- **base_t1**: the answer must be inferred. The earlier table is a randomly drawn group table (Z6 or S3; the same type as the target in 53 of 100 trials), written in six *other* words, so no target line is present.

Behaviour confirms the difference. Accuracy is 0.53–0.97 in leak and 0.17–0.22 in base (chance 0.17).

**Implementation.** We reimplemented the published detector:
- its `_format_prompt`;
- the 30 + 30 calibration prompts;
- the mean-difference axis;
- the best-of-four layer;
- the last-token decoder-layer output.

On uncached forwards it matches `CRDetector._get_hidden` exactly (max relative difference 0, Qwen3-0.6B-Base).

**Differences from the published setup.**
- bf16 instead of fp16.
- 4B and 8B in nf4, with 11–12 of 26 queries per trial; 8B with its embeddings on the CPU.
- A shared-prefix KV cache for Experiment 5 (r ≥ 0.996 against uncached forwards). Experiment 5b uses uncached forwards.

The direction (base projects toward "compute") was preregistered. It was finalised minutes after a discarded 6-trial smoke test whose estimate pointed against it.

## 2. Result of the preregistered test (Experiment 5)
- **Calibration.** It separates well on these five small Qwen3 models: AUC 0.96–0.99 at the chosen layer. The paper tested Qwen2.5 0.5–7B and Qwen3-30B-A3B. Three of our five models (0.964–0.978) are slightly below the paper's ≥0.988.
- **H1 (base > leak) is supported in 0 of 5 models.** Detector format (chat template), detector layer:

| model | base − leak [95% CI] | d | query AUC (base > leak) |
|---|---|---|---|
| Qwen3-0.6B-Base | −27.2 [−32.7, −21.5] | −1.34 | 0.19 |
| Qwen3-1.7B-Base | −158.5 [−173.0, −144.5] | −3.08 | 0.03 |
| Qwen3-1.7B (post-trained) | −0.06 [−1.5, +1.6] | −0.01 | 0.50 |
| Qwen3-4B-Base (nf4) | −11.1 [−15.5, −6.5] | −0.67 | 0.34 |
| Qwen3-8B-Base (nf4) | +0.70 [−6.7, +8.0] | +0.03 | 0.51 |

- **Three base models are significantly reversed.**
- **The direction depends on the wrapper.** In raw format (no chat template, exploratory), base > leak with the CI excluding 0 in 0.6B-Base, 1.7B-Base and 4B. That would meet the ≥3-of-5 criterion had raw format been primary. Base < leak in 1.7B (d = −0.28) and 8B (d = −1.68).
- **No graded effect.** Within base, projection does not rise with inference demand (L < O < G). ρ is −0.003 to −0.11, and significantly negative in 1.7B-Base.
- **Blind to correctness within a condition.** AUC is 0.46–0.55 (0.38 in 1.7B-Base leak). This is consistent with the paper's finding that the axis is blind to correctness.

## 3. Why: context vocabulary dominates, and the retrieval signal appears only once it is matched (Experiment 5b)
Leak and base are matched on length and format, but not on vocabulary:

| | leak | base |
|---|---|---|
| distinct content words | 6 | 12 |
| query words' occurrences in the context | 45.5 | 9.5 |
| repeated 4-gram token fraction | 0.48 | 0.15 |
| query line present verbatim | always | never |

In detector format the last token is the template's newline after `assistant`, so the projection summarises the whole context.

**Two relabelled conditions** (100 trials, all 26 queries, uncached; only the history table changes):
- **leakRel**: leak's history with its words deranged. The vocabulary, word counts and query-word occurrences are identical to leak, but the answers are inconsistent. This is the cleanest retrievability contrast: **leakRel − leak** (predicted > 0).
- **baseRel**: base's history written in the target's words. It is not retrievable. By chance, 11% of the target's ten displayed lines appear verbatim in its history table (8% for leakRel), and 13% of its 36 lines agree with the target's full table (9% for leakRel). **f_vocab = (baseRel − base)/(leak − base)** is the share of the base–leak gap reproduced by vocabulary alone.

At the detector layer:

| model | format | base − leak (d) | f_vocab [CI] | leakRel − leak [CI] (paired d_z) |
|---|---|---|---|---|
| Qwen3-0.6B-Base | detector | −27.3 (−1.34) | 0.67 [0.54, 0.85] | −4.4 [−6.7, −2.1] (−0.36) |
| Qwen3-1.7B-Base | detector | −158.6 (−3.08) | 0.87 [0.79, 0.96] | −11.8 [−20.0, −3.5] (−0.28) |
| Qwen3-1.7B | detector | −0.07 (−0.01) | n/a (no gap) | **+1.3 [+0.6, +2.1] (+0.36)** |
| Qwen3-4B-Base (nf4) | detector | −10.5 (−0.64) | 0.64 [0.40, 0.99] | −1.1 [−3.2, +1.0] (−0.10) |
| Qwen3-0.6B-Base | raw | +5.8 (+0.79) | 0.68 [0.48, 1.04] | **+1.2 [+0.8, +1.6] (+0.60)** |
| Qwen3-1.7B-Base | raw | +12.0 (+0.44) | (undefined: vocabulary moves the other way) | **+21.7 [+19.2, +24.2] (+1.67)** |
| Qwen3-1.7B | raw | −1.1 (−0.28) | — | **+1.8 [+1.3, +2.3] (+0.69)** |
| Qwen3-4B-Base (nf4) | raw | +4.9 (+0.77) | 0.74 [0.53, 1.18] | **+1.0 [+0.4, +1.5] (+0.33)** |

**Reading** (exploratory):
1. **In the base models with the chat template, vocabulary overlap explains most of the reversed gap** (64–87%).
   - The remainder (leak − baseRel) is still reversed in all three: 0.6B-Base +8.9 [+3.5, +14.5], 1.7B-Base +20.6 [+5.2, +35.8], 4B-Base +3.8 [+0.1, +7.4].
   - With the vocabulary held fixed (leakRel − leak), the retrievable context projects *more* toward compute in 0.6B-Base and 1.7B-Base, and no differently in 4B-Base.
2. **With the vocabulary held fixed, the predicted direction appears in two cases:** the post-trained model, in both formats (d_z +0.36 and +0.69), and all four models in raw format at the detector layer (d_z +0.33 to +1.67; at other layers the sign varies). Making the retrievable answer inconsistent pushes the projection toward "compute", as the paper's interpretation predicts.
3. **In the post-trained model the two effects cancel.** Vocabulary overlap raises the projection (baseRel − base = +1.8), and retrievability lowers it by about the same amount. The preregistered base − leak comparison therefore shows nothing (d = −0.01).

So the axis does carry an in-context retrieval signal of the right sign:
- in the post-trained model it is small (d_z ≈ 0.4–0.7, against a calibration separation of AUC ≈ 0.99), about the size of the vocabulary effect;
- in raw format it ranges from d_z +0.33 to +1.67.

In base models under the chat template, the vocabulary effect is 4–12× larger, and the retrieval contrast itself reverses.

## 4. What this does and does not show
- **It does not refute the paper's main claim.** The paper's "retrieve" is *parametric* recall: facts stored in weights. Here, "retrieve" is *in-context* lookup. The results bound how the axis generalises. They do not show that the calibration separation is wrong.
- **The pattern agrees with the paper's own observation** that base models do not separate matched-format pairs (Pythia-410M: 3/20) while instruction-tuned models do. Here, the post-trained model shows the predicted retrievability effect under the detector's own (chat-template) format, and the base models do not.
- **It does show strong non-mode sensitivities.** Under exact length and format control:
  - the projection moves by up to 3 SD with context vocabulary and repetition;
  - in base models, its sign depends on the prompt wrapper.

  Any use of the axis as a mode detector on a new prompt distribution should control for both.

## 5. Possible follow-ups for the CR-axis preprint
1. **Limitations.** A suggested paragraph:
   > *On a format- and length-matched contrast in which the same query is answered either by in-context lookup or by inference (Cayley-table completion; this study's compounding-fingerprint stimuli), the CR axis, applied as published, did not separate the conditions in the predicted direction in any of five Qwen3 models (0.6B–8B). It reversed significantly in three base models, and the sign depended on whether the chat template was applied. An exploratory control showed that differences in context vocabulary explain most of the reversal (64–87%). With vocabulary held fixed, the axis moved in the predicted direction in the instruction-tuned model (d_z ≈ 0.4) and, without the chat template, in all four models tested in the control (0.6B–4B; 8B was not run). The axis therefore carries a retrieval signal, but one that context vocabulary and repetition can match or exceed. Claims about a process-level "mode" on new prompt distributions need matched-vocabulary controls.*
2. **A repetition-matched control.** Add pairs in which retrieval prompts repeat context material and compute prompts do not, and the reverse.
3. **Report sign stability across prompt wrappers** (chat template vs raw) for every format control, especially for base models.
4. **Temper "measures computational demand"** to "separates prompt types that differ in computational demand". The conclusion's open question ("a deep computational mode or a shallower proxy for prompt complexity") can cite this as evidence for both: a real but small mode signal, under a larger surface component.
5. **Optional new experiment.** The fingerprint stimuli, `exp5/cr_probe.py` and `exp5/ctrl_vocab.py` provide a ready out-of-distribution test. The most informative next run is an instruction-tuned model in fp16, since that is where the paper's format controls pass (Gemma 4 E2B, Qwen2.5-0.5B-Instruct, LLaMA-3.2-1B-Instruct). Ideally it would be 7B or more, a scale at which the paper ran no format control.
