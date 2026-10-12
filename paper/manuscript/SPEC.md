# Manuscript specification (for every writer and checker)

*Drafting brief used to assemble the manuscript with AI assistance (see the Statements section of the paper). It is kept as provenance and as the reference for the section syntax that `build.py` accepts.*

The paper is written once, as Markdown section files in `paper/manuscript/sections/`. `build.py` turns them into `main.tex` (arXiv/Overleaf) and `paper.html` → `paper.pdf`. **Use only the syntax below**; the builder rejects anything else.

## Working title
*A Compounding Fingerprint: Telling Abstraction from Exemplar Reuse in Context*
(Subtitle-level claims belong in the abstract, not the title.)

## Sources of truth (numbers MUST come from these; never invent, round consistently, keep CIs as given)
| content | file |
|---|---|
| main table, every learner/model/network | `paper/results_tables.md` |
| calibration | `paper/cal_summary.md`, `exp2/RESULTS2_tables.md`, `exp2/RESULTS_2d.md` |
| scoring checks | `exp2/offchk_result.md` |
| count shortcut | `exp3/heur/count_heuristic.md`, `exp3/heur/intrain_heuristic.json` |
| CR axis | `exp5/RESULTS_tables.md`, `exp5/RESULTS_ctrl.md`, `exp5/CR_PAPER_NOTE.md` |
| positive control | `exp6/RESULTS.md`, `exp6/RESULTS_tables.md`, `exp6/mechanism.md`, `exp6/validate_*.json` |
| prose drafts (bullet style; audited except where noted below) | `paper/abstract.md`, `paper/methods.md`, `paper/results.md`, `paper/discussion.md`, `paper/related_work.md`, `paper/positioning.md` |
| preregistrations | `exp1/PREREGISTRATION.md`, `exp2/PREREGISTRATION_ADDENDUM.md`, `exp3/PREREGISTRATION.md`, `exp5/PREREGISTRATION.md`, `exp6/PREREGISTRATION.md` |

The drafts in `paper/*.md` are the audited content. Your job is to turn them into **paper prose** (paragraphs, not bullet lists) without changing any number, hedge or label. Where a draft and a generated table disagree, the generated table wins; say so in your final report.

Do **not** include material the audits never covered. In particular, leave out the "minimum defining sets" and "⌈(n−1)/3⌉ lower bound" claims in `related_work.md`/`positioning.md`, which come from an unaudited earlier report.

## Voice and style
- Use "we". Past tense for what was done ("we scored"), present tense for what results show ("the rule returns").
- Write paragraphs. Bullets are allowed only for the contribution list in the Introduction and for short enumerations of three or more parallel items.
- Be precise and unhyped. No "groundbreaking", "novel paradigm", "remarkably" and the like.
- Keep every status label that appears in the sources: *preregistered*, *exploratory*, *post hoc*, *not met*.
- Each paragraph should make one point.
- Terms (use exactly these, consistently):
  - **compounding fingerprint**; **forcing class** (L, O, G, Ab, Cy, U).
  - "Latin-forced (L)", "loop-forced (O)", "associativity-forced / G-forced (G)".
  - learner classes **R, N, N_iso, A, B-type**; idealised learners R, N, N_iso, A, Asl, B, Bh, and post-hoc **C** (count shortcut).
  - **history** (earlier complete tables, t of them), **cue** (k revealed target cells), **leak control**, **scramble control**, **held-out types**.
  - acc_L, g(G t0), comp, held, w (see math below).
  - The pretrained-model set: "nine checkpoints (ten model–precision configurations) from three families".
  - The positive-control network: "the released In-Context Algebra model".
  - The axis: "the compute–retrieve (CR) axis".
- Numbers: two or three decimals as in the sources. Use the Unicode minus "−" for negative numbers, "×" for products, and "–" for ranges. Write 95% CIs as `[lo, hi]`.

## Syntax (strict)
- **Section file header.** Each file starts with `## Title {#sec:label}`. Subsections use `### Title {#sec:label}`, and paragraph heads use `#### Title`. The builder numbers everything.
- **Paragraphs** are separated by blank lines.
- **Inline formatting**: `**bold**`, `*italic*`, `` `code` `` (file names and code identifiers only).
- **Lists** use `- item` or `1. item`, with no nesting.
- **Citations** use `[@key]` or `[@key1; @key2]`, with keys only from `refs.json`. To introduce a new key, add a line to your final report; do not edit `refs.json` unless you are the citation checker.
- **Cross-references.** Write `@sec:label`, `@fig:label`, `@tab:label`, `@app:label` inline. The builder renders "Section 3.2", "Figure 4", "Table 2", "Appendix B". Do not type the words "Section", "Figure" or "Table" before a reference.
- **Figures.** One line, on its own:
  `![Caption text, one or two sentences, ending with a period.](fig/<file>.png){#fig:label}`
- **Tables.** A caption line, a blank line, then a pipe table with a header row and a `|---|` separator:
  `Table: Caption text. {#tab:label}`
  Keep tables at seven columns or fewer, with short cell text.
- **Math** goes in `$...$`, with only this subset:
  - letters, digits, `+ - = < > ( ) [ ] , . /`;
  - `_{...}` and `^{...}`;
  - `\geq \leq \times \approx \cdot \ldots \to \in \neq \pm \infty \lfloor \rfloor \lceil \rceil`;
  - Greek `\alpha \beta \gamma \delta \rho \sigma \theta \mu`;
  - `\mathrm{...}`, `\text{...}`, `\frac{a}{b}`.

  Prefer plain text where math is not needed, e.g. "acc_L" can be written `$\mathrm{acc}_L$`.
- **Forbidden**: raw HTML or LaTeX outside `$...$`, footnotes, nested lists, blockquotes, images other than the six figures, and links in the body. URLs belong in `refs.json` or in the Statements section.

## Fixed labels (use these; do not invent others without listing them in your final report)
**Sections**
- `sec:intro`, `sec:related`, `sec:fingerprint`;
- fingerprint subsections: `sec:task`, `sec:classes`, `sec:conditions`, `sec:learners`, `sec:rule`;
- `sec:setup`, with `sec:models`, `sec:calib`, `sec:trained`, `sec:probe`, `sec:cr`, `sec:ica`, `sec:stats`;
- `sec:results`, with `sec:r-ideal`, `sec:r-llm`, `sec:r-cal`, `sec:r-trained`, `sec:r-probe`, `sec:r-cr`, `sec:r-ica`;
- `sec:discussion`, `sec:conclusion`.

**Appendices** (the file `A_appendix.md`, each starting `## Title {#app:label}`): `app:main`, `app:calib`, `app:trained`, `app:cr`, `app:ica`, `app:deviations`.

**Figures** (all six go in the Results section, near their subsections):

| label | file | content |
|---|---|---|
| `fig:ideal` | `fig/fig1_ideal_learners.png` | |
| `fig:llm` | `fig/fig2_llm_contrasts.png` | |
| `fig:cal` | `fig/fig3_calibration.png` | |
| `fig:trained` | `fig/fig4_trained_transformers.png` | |
| `fig:cr` | `fig/fig5_cr_axis_control.png` | the 5b control, in SD units |
| `fig:ica` | `fig/fig6_in_context_algebra.png` | |

**Tables**: `tab:conditions`, `tab:rule`, `tab:models`, `tab:learners`, `tab:recall`, `tab:cal`, `tab:trained`, `tab:cr`, `tab:ica`, plus appendix tables `tab:app-*`.

## Section owners and target lengths (words, body text)

| file | content | owner | target |
|---|---|---|---|
| `00_abstract.md` | one paragraph, no citations, no cross-refs; starts `## Abstract {#sec:abstract}` | W1 | ≤ 250 |
| `01_introduction.md` | | W1 | 1,000–1,300 |
| `02_related_work.md` | | W2 | 900–1,200 |
| `03_fingerprint.md` | the method | W3 | 1,000–1,300 |
| `04_setup.md` | models, scoring, calibration, trained networks, probe, CR axis, positive control, statistics, preregistration and audits | W3 | 1,200–1,600 |
| `05_results.md` | R1–R7 as subsections, figures and main-text tables | W4 | 2,800–3,600 |
| `06_discussion.md` | | W5 | 1,100–1,500 |
| `07_conclusion.md` | | W1 | 200–300 |
| `A_appendix.md` | | W5 | as needed |

`08_statements.md` (AI-assistance disclosure, preregistration, reproducibility, data availability) is written by the coordinator.

## Key story (all writers align to this)
1. **Question.** When a model gets better at a structured task after seeing earlier examples in context, did it abstract the structure, or reuse the examples? Behavioural improvement alone cannot tell. Length, recall and exemplar matching all produce it.
2. **Method.** The *compounding fingerprint* labels every queried cell of a partial Cayley table by the weakest axiom class that forces it (exactly, by enumeration or SAT). It controls the history with length-matched scramble, held-out isomorphism types, a leak (recall) control and non-associative loops. A preregistered decision rule maps the signature to learner classes. Seven exact idealised learners are recovered with 0 errors. Abstraction and exemplar matching differ only in held-out transfer.
3. **Pretrained LMs.** Nine checkpoints (ten model–precision configurations) from three families (Qwen3 0.6–8B base and post-trained, SmolLM2-1.7B, OLMo-2 1B and 7B) are all class R. There is no compounding (t4 − scramble −0.019 to +0.026, every CI includes 0). Recall is family-dependent: Qwen3 reaches 0.95–0.96 at 4–8B. Direct-answer Latin inference is absent, even at dense cues where row/column elimination would score 0.92.
4. **Trained networks.** Small transformers trained from scratch learn a global count shortcut, which post-hoc learner C matches to 0.001–0.007. The fingerprint classifies them correctly, and the preregistered learning-failure threshold cannot tell the shortcut from the task.
5. **Positive control.** The released In-Context Algebra model is non-R, non-B: L/O cells 0.89–0.99, G cells 0.53–0.70, with a partial group prior and a commutative-copying shortcut. Above the loop level at n=6 and n=8, not at n=5. The class changes during training (exploratory). Its 16-token vocabulary leaves the compounding branch untested.
6. **CR axis.** A published activation "compute vs retrieve" axis fails a format- and length-matched retrieval/inference contrast (0 of 5, reversed in 3). Context vocabulary explains 64–87% of the reversal. With vocabulary matched, a retrieval signal of the predicted sign appears in the post-trained model and without the chat template.
7. **Reasoning probe.** Inconclusive, because truncation dominated.
8. **Takeaways.** Claims of in-context abstraction need length-matched controls, structure-specific cells, held-out types and a count baseline. Activation "mode" probes need format- and vocabulary-matched contrasts. All of this ran on one 8 GB laptop GPU, with every deviation from preregistration disclosed (Appendix `app:deviations`).

## Writers' final report (your return value)
Return a short report:
- the files you wrote;
- word counts;
- any new citation keys or labels you need;
- every place where you could not find a number in the sources, or where sources disagreed.
