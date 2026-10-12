# Paper materials: index

**Title:** *A Compounding Fingerprint: Telling Abstraction from Exemplar Reuse in Context*

## Assembled manuscript
The full paper is in `manuscript/`: `paper.pdf`, `main.tex` and `arxiv_upload.zip`, built from `manuscript/sections/*.md` (see `manuscript/README.md`). The files below are the pre-manuscript drafts it was written from. Where they differ, the manuscript is authoritative.

## Text
| file | content |
|---|---|
| `abstract.md` | abstract and contribution list |
| `positioning.md`, `related_work.md` | positioning notes and a related-work survey (drafts written 2026-10-07, after the Experiment-1 results; they use no Experiment-2 output). They were only partly verified and contain unaudited minimum-defining-set claims that the paper leaves out. |
| `methods.md` | task, forcing classes, learners, decision rule, models, scoring, training, Experiments 4–6, **deviations register** |
| `results.md` | R1 to R7 |
| `discussion.md` | what is and is not established, threats to validity, implications |

## Tables (auto-generated; re-run the scripts to regenerate)
| file | script | content |
|---|---|---|
| `results_tables.md`, `results_all.json` | `collect.py` | every learner, model and network on the identical 1,800 trials: classes, contrasts with CIs, derived quantities |
| `cal_summary.md`, `cal_summary.json` | `cal_summary.py` | calibration: L accuracy by k, k28−k10, sorted−raw and instructed−raw contrasts |
| `../exp2/RESULTS2_tables.md` | `../exp2/analyze2.py` | per-model cue-sweep detail |
| `../exp2/RESULTS_2d.md` | `../exp2/analyze_2d.py` (stimuli: `../exp2/make_stim_k16.py`) | addendum-2d k=16 operating point for Qwen3-0.6B/1.7B-Base |
| `../exp2/offchk_result.md` | `../exp2/offchk_compare.py` | scoring-path checks: offload vs standard; the re-scored primary contrast |
| `../exp3/heur/count_heuristic.md` | `../exp3/heur/count_heuristic.py` | count-shortcut diagnostic (post hoc) |
| `../exp5/RESULTS_tables.md`, `../exp5/RESULTS_ctrl.md` | `../exp5/analyze5.py`, `../exp5/analyze_ctrl.py` | CR axis: preregistered H1/H2, and the vocabulary control (5b) |
| `../exp6/RESULTS_tables.md`, `../exp6/mechanism.md` | `../exp6/analyze6.py`, `../exp6/mechanism.py` | In-Context Algebra positive control (Experiment 6); `../exp6/RESULTS.md` is a hand-written summary of these tables |

## Figures (`fig/`, PNG and PDF; data in `fig/figure_data.json`; script `figures.py`)
1. `fig1_ideal_learners`: the exact learners' signatures (G accuracy vs t, held-out).
2. `fig2_llm_contrasts`: t4−scramble and held-out−scramble per pretrained model, with CIs.
3. `fig3_calibration`: L accuracy vs cue density per model.
4. `fig4_trained_transformers`: trained networks beside the ideal A, N_iso and C learners.
5. `fig5_cr_axis_control`: CR projection per condition minus leak, in units of the pooled SD of leak/base trial means, at the detector format and layer. The data come from the Experiment-5b run: leak and base re-measured uncached with all 26 queries, four models, 8B not run.
6. `fig6_in_context_algebra`: (a) L/O/G accuracy at n=6 base t=0 for the In-Context Algebra model, Qwen3-8B-Base, A4 and idealised learners; (b) cue sweep vs learner C and row/column elimination.

## Preregistrations
Each was frozen before the outputs it governs, with two exceptions: the Experiment-5 file was finalised minutes after a discarded 6-trial smoke test on a subset of its data, and the ≥30-query guard on the Experiment-2 k_L rule was appended after the first calibration sweep had been seen. Later notes are appended, never edited in.
`../exp1/PREREGISTRATION.md`, `../exp2/PREREGISTRATION_ADDENDUM.md`, `../exp3/PREREGISTRATION.md`, `../exp5/PREREGISTRATION.md`, `../exp6/PREREGISTRATION.md`. The exploratory designs are `../exp4/NOTE.md` and `../exp5/CONTROL_vocab.md`.

## Audits
`../audit/` (first) and `../audit2/` (final) hold the independent re-derivation scripts. Every finding of the final audit is addressed in the text. Three more checks were run in response:
- offload equivalence, now archived;
- the primary contrast re-scored with one code version;
- the vocabulary control (Experiment 5b).

A verification pass over the revised text then confirmed 51 further discrepancies. All were corrected, including running the addendum-2d k=16 step, which the post-hoc ≥30-query guard had excluded.

## Reproduction
- CPU analyses: follow "Reproducing the analyses" in `../README.md`, which runs every analysis script in order. `../exp6/loop_level.py` recomputes the exact loop-level baseline separately (it needs `python-sat`).
- GPU runs:
  - `../finish_all.py` completes unfinished main runs. It does not re-create the Experiment-1 runs (`../exp1/run.py`) or the Qwen3-4B/8B-Base and OLMo-2 calibration files (scored with `../exp2/run.py` from `../master.sh`, `../master2.sh` and `../exp2/jobs*.txt`).
  - `../gpu_checks.py` runs the post-audit checks, and `../gpu_2d.py` the addendum-2d k=16 follow-up.
  - Experiment 6 runs through `../exp6/gpy.sh`: `validate.py`, `validate_unshuffled.py`, `validate_loss.py` and `validate_ckpts.py`, then `score.py` for the main, k16, cal, interleaved and checkpoint runs (commands in `../exp6/PREREGISTRATION.md` and `../exp6/RESULTS.md`).
  - LLM scoring and training resume at trial or checkpoint level. The Experiment-5 projections (`cr_probe.py`) restart per model.
