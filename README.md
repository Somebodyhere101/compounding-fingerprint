# A Compounding Fingerprint: Telling Abstraction from Exemplar Reuse in Context

Code, stimuli, raw results, preregistrations, audits and manuscript for the paper of the same name. **This project was carried out with substantial AI assistance** (see [Use of AI assistance](#use-of-ai-assistance)). The paper is in [`paper/manuscript/paper.pdf`](paper/manuscript/paper.pdf). It is built from `paper/manuscript/sections/*.md`, and the generated LaTeX is `paper/manuscript/main.tex`.

**The question.** When a model gets better at a structured task after seeing earlier examples in context, did it abstract the structure, or reuse the examples?

**The method.** Models complete partial Cayley tables, written in fresh word labels, after 0–4 earlier complete tables (the *history*).
- Every queried cell is labelled exactly, by enumeration or SAT, with the weakest axiom class that forces it: Latin (L), loop (O) or group-only (G).
- Histories are controlled by a length-matched scramble, held-out isomorphism types, a recall (leak) control and non-associative loops.
- A preregistered decision rule maps the resulting signature to learner classes, and recovers seven exact idealised learners with no error.

## Main results

| | finding | where |
|---|---|---|
| Pretrained LMs | Nine checkpoints (ten model–precision configurations) from Qwen3, SmolLM2 and OLMo-2, 0.6–8B, are all recall-only (class R). None compounds: t=4 minus scramble is −0.019 to +0.026, and every CI includes 0. | Exp. 1–2 |
| Trained networks | Small transformers trained from scratch learn a global count shortcut, which a post-hoc count learner matches to within 0.001–0.007 in five of six learned networks. | Exp. 3 |
| Positive control | The released In-Context Algebra model is classified non-R, non-B. At t=0, Latin- and loop-forced cells score 0.89–0.99 and G cells 0.53–0.70. | Exp. 6 |
| CR axis | The author's earlier "compute vs retrieve" activation axis ([preprint](https://doi.org/10.5281/zenodo.19455724)) moves significantly in the predicted direction in 0 of 5 models on a format-matched retrieval/inference contrast, and is significantly reversed in 3. | Exp. 5, 5b |
| Reasoning probe | Inconclusive: truncation dominated. | Exp. 4 |

All model inference and training ran on one 8 GB laptop GPU (RTX 5070).

## Repository layout

| path | contents |
|---|---|
| `runs/c10/` | Design phase (2026-10-06): the stimulus generator `code/stimuli.py`, the exact forcing-class engine `code/engine.py`, group, loop and Latin tables, and the design report and notebook (see `runs/c10/README.md`). |
| `exp1/` | Main stimuli (`stim/`: 18 cells × 100 trials), the preregistration, the LM scorer `run.py`, the decision rule `analyze.py`, the idealised learners (`ideal/`) and the results of the first three LMs. |
| `exp2/` | Preregistration addendum: calibration cue sweep (`stim/`), more models up to 8B, the quantisation control and the addendum-2d k=16 step (`stim_k16/`). It also holds the scorer with CPU offload and nf4, and the scoring-path checks. `run_v1_backup.py` is the earlier scorer version that produced the v1-scored records compared in `audit2/quant.py`. |
| `exp3/` | From-scratch transformers: `data.py`, `model.py`, `train.py`, `eval_fp.py`, the preregistration, training logs (`runs/*/log.csv`, `eval.csv`, `config.json`), fingerprint results (`res/`), and the post-hoc count-shortcut diagnostic (`heur/`). `train_v1_backup.py` is `train.py` before its time cap was changed to count only active compute time. |
| `exp4/` | Exploratory reasoning probe (`cot.py`, `NOTE.md`, results). |
| `exp5/` | CR-axis experiment: `cr_probe.py`, the preregistration, `analyze5.py`, and the exploratory vocabulary control 5b (`ctrl_vocab.py`, `CONTROL_vocab.md`, `analyze_ctrl.py`). `CR_PAPER_NOTE.md` summarises the implications for the CR preprint. |
| `exp6/` | Positive control: re-implementation of the released In-Context Algebra model (`ica.py`), validation scripts, scorer `score.py`, `analyze6.py`, `mechanism.py`, `loop_level.py`, the preregistration and results. |
| `audit/`, `audit2/` | Scripts and outputs of the two adversarial audits that re-derived the reported numbers independently. |
| `paper/` | Pre-manuscript drafts (see `paper/README.md`), generated tables (`collect.py`, `cal_summary.py`), figures (`figures.py` writes `fig/`), and the assembled manuscript in `manuscript/` (see `paper/manuscript/README.md`). |
| `*.sh`, `*.py`, `*.log` at the top level | The GPU job queues and finishers that ran everything (`master.sh`, `master2.sh`, `after_master*.sh`, `finish_all.py`, `gpu_checks.py`, `gpu_2d.py`) and their logs. The logs (`master2.log`, `finish_all.log`, `gpu_checks.log`, `gpu_2d.log`) are the provenance for the dates and times in the deviation register. `keepawake.py` keeps the laptop from sleeping while a queue runs. |

Every deviation from the preregistrations is listed in Appendix G of the paper and in `paper/methods.md`.

## Reproducing the analyses (CPU, from the committed results)
1. Clone the repository and enter it:

   ```
   git clone https://github.com/Somebodyhere101/compounding-fingerprint.git
   cd compounding-fingerprint
   ```

2. Create a Python 3.12 environment and run `pip install -r requirements-cpu.txt`. On Windows, also turn on Python's UTF-8 mode (`export PYTHONUTF8=1` in Git Bash, `$env:PYTHONUTF8=1` in PowerShell, `set PYTHONUTF8=1` in cmd). Without it, scripts whose output is not a console stop with a `UnicodeEncodeError` before they write anything.
3. Run the analysis scripts, each from the folder it lives in:

   ```
   cd exp1 && python analyze.py && cd ..
   cd exp1/ideal/code && python validate.py && cd ../../..
   cd exp2 && python analyze2.py && python analyze_2d.py && python offchk_compare.py && python k28_elimination.py && cd ..
   cd exp3 && python analyze3.py && cd ..
   cd exp3/heur && python count_heuristic.py && cd ../..
   cd exp5 && python analyze5.py && python analyze_ctrl.py && cd ..
   cd exp6 && python analyze6.py && python mechanism.py && cd ..
   cd paper && python collect.py && python cal_summary.py && python figures.py && cd ..
   cd paper/manuscript && python build.py && cd ../..
   ```

On Windows with the pinned packages and MiKTeX 25.12, each step regenerates its committed outputs byte for byte, including `paper.pdf`, so `git status` stays clean. On other systems this was not tested: the numbers and text should match, but figure files and `paper.pdf` may differ in their bytes. `exp6/loop_level.py` recomputes the exact loop-level baseline. It needs `python-sat` and the engine in `runs/c10/code/`, and it is resumable: delete `exp6/res/loop_level.jsonl` to recompute from scratch. `paper/manuscript/build.py` compiles `paper.pdf` from the arXiv bundle with pdfLaTeX (MiKTeX or TeX Live) when one is installed, and otherwise prints `paper.html` with a local Chrome, Chromium or Edge; use `--no-pdf` to skip that step.

## Re-running the GPU experiments
The GPU scripts were run on Windows and contain local paths. Edit them before re-running:
- `GPY` (GPU Python) in `finish_all.py`, `gpu_checks.py`, `gpu_2d.py`, `master.sh`, `master2.sh`, `after_master2.sh`, `exp2/gpu_queue*.sh`, `exp3/queue3.sh`, `exp3/g4fix.sh`, `exp4/run_cot2.sh` and `exp5/run_exp5.sh`; `after_master.sh` calls the GPU Python by its full path;
- `CPY` (CPU Python) in `finish_all.py` and `gpu_2d.py`; `exp3/queue3.sh`, `exp3/g4fix.sh`, `exp4/run_cot2.sh` and `exp5/run_exp5.sh` call the CPU Python by its full path;
- `LOCK` in `master.sh` and `exp2/gpu_queue*.sh`, `PY` in `exp1/ideal/code/run_shards.sh`, and the interpreter and `PYTHONPATH` in `exp6/gpy.sh`.

`after_master*.sh` wait for the end of a `master.sh` run, recorded in the file named by `MASTER_OUT`, and are kept as provenance. The LM scorers read checkpoints from `HUB = ~/.cache/huggingface/hub`, not from `HF_HOME` or `HF_HUB_CACHE`.

- **Experiments 1–2** (`exp1/run.py`, `exp2/run.py`) need the Hugging Face checkpoints Qwen/Qwen3-{0.6B,1.7B}(-Base), Qwen/Qwen3-{4B,8B}-Base, HuggingFaceTB/SmolLM2-1.7B-Instruct, allenai/OLMo-2-0425-1B and allenai/OLMo-2-1124-7B in the local HF cache. They run offline.
- **Experiment 3.** `exp3/train.py`, then `exp3/eval_fp.py`. The checkpoints are not versioned; see below.
- **Experiment 4** (`exp4/cot.py`, run by `exp4/run_cot2.sh`) needs Qwen/Qwen3-1.7B.
- **Experiment 5** imports `cr_axis.detector` from the CR-axis code, <https://github.com/Somebodyhere101/compute-retrieve> at commit `ce0c61f` (the code accompanying the preprint, doi:10.5281/zenodo.19455724). Set `UP` in `exp5/cr_probe.py`, `exp5/ctrl_vocab.py`, `exp5/ctrl_vocab_unbatched.py` and `audit2/e5_gpu*.py`, and the `sys.path.insert` line in `audit2/e5_ctrl.py`, to a local clone.
- **Experiment 6** needs the released In-Context Algebra checkpoints in `exp6/dl/weights/`, downloaded from <https://algebra.baulab.info/weights/mixrosette-facts-10-16-8heads/models/>. Fetch `algebra_gpt_{0,5000,10000,20000,40000,100000,best}.pt` and check them with `cd exp6/dl/weights && sha256sum -c ../weights_SHA256SUMS`. The authors' code (<https://github.com/ericwtodd/algebra>, commit 532a6a2) was read but not executed; `exp6/ica.py` re-implements it.
- **Idealised learners** (`exp1/ideal/code/ideal.py`) need the small C helper `latc.c` compiled as a shared library next to it, for example `gcc -O2 -shared -o latc.dll latc.c` on Windows. On Linux or macOS, build it with `gcc -O2 -shared -fPIC "-D__declspec(x)=" -o latc.so latc.c` and change `latc.dll` to `latc.so` in `cnt.py` (line 6).

## Notes on the record
- **Local paths.** Before publication, absolute local paths in scripts, logs and run configs were replaced by placeholders, and nothing else in those files was changed. `<repo>` stands for the repository root at run time. In the shell scripts the Windows account name is `$USERNAME`, which Git Bash expands; elsewhere it is `%USERNAME%`. The path to the CR-axis code is a placeholder. The frozen `exp5/PREREGISTRATION.md` still names the relative local folder where the author read that code; the public code is <https://github.com/Somebodyhere101/compute-retrieve>. Some working notes (`exp3/README.md`, `runs/c10/`) also had their paths replaced and were lightly reworded for outside readers.
- **Wording in the working notes.** The preregistrations and working notes were written by the AI assistant during the project. In them, "the user" means the author, and "the brief" means the author's initial task description, which is not included.

## Not in this repository
- **Released In-Context Algebra weights.** These are seven files of 212 MB each; re-download them and verify the checksums.
- **Third-party code.** The In-Context Algebra code (ericwtodd/algebra, MIT License) is not included. `exp6/dl/metadata.json` is the training configuration published with the released checkpoints, kept as provenance (see `exp6/dl/README.md`).
- **Trained-network checkpoints** (`exp3/runs/*/*.pt`, about 585 MB). The training logs and evaluation results are included, and `exp3/train.py` retrains them.
- **Intermediate shards** of the idealised-learner runs (`exp1/ideal/parts/`). The merged results are in `exp1/ideal/res/`.

## Use of AI assistance
Substantial AI assistance was used throughout this project. An AI system (Claude, Anthropic), working under the author's direction, contributed to the study design, wrote the code and the preregistrations, ran the experiments, analyses and audits, and drafted the paper and the notes in this repository. See the Statements section of the paper.

## License and citation
- **Code** (every `.py`, `.sh` and `.c` file, including those under `paper/`) is under the MIT License; see `LICENSE`.
- **Everything else** (the paper text and figures, the stimuli, the result data, generated tables and summaries, and the notes) is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Copyright (c) 2026 Sam Ramdan.
- **Third-party material** keeps its own terms: `exp6/dl/metadata.json` belongs to the In-Context Algebra authors (see `exp6/dl/README.md`). `exp6/ica.py` re-implements parts of their code (ericwtodd/algebra, MIT License, Copyright (c) 2025 Eric Todd); see its docstring.
- **Citation.** To cite this work, use `CITATION.cff`; GitHub's "Cite this repository" button reads it.
