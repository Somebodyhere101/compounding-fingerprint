## Statements {#sec:statements}

#### Preregistration
Experiments 1, 2, 3, 5 and 6 were preregistered in dated files frozen before the outputs they govern: 2026-10-06, 2026-10-07, 2026-10-07, 2026-10-08 and 2026-10-11. The exceptions are Experiment 5, whose file was finalised minutes after a discarded six-trial smoke test, and the $\geq 30$-query guard on the Experiment 2 $k_L$ rule, appended after the first calibration sweep had been seen (@app:deviations, row 4). Experiment 4 and Experiment 5b were exploratory designs written down before their final runs: Experiment 4 was redesigned after its first run was truncated, and Experiment 5b followed a 20-trial audit pilot of the same design (@sec:probe; @sec:cr). Later notes were appended to these files, never edited in. Every deviation is listed in @app:deviations.

#### Audits
Two adversarial audits (2026-10-07 and 2026-10-09) re-derived every Experiment 1–5 number that existed at the time from the raw result files, using independent code. The results produced after the second audit (learner C, the count-shortcut diagnostic, the offload and re-scoring checks, the full Experiment 5b run and the addendum-2d run) were re-derived instead by later verification passes (2026-10-09), and further passes checked Experiment 6 and the assembled manuscript (2026-10-11). Each confirmed discrepancy was corrected, and the corrections that changed a reported number or verdict are listed in @sec:discussion and @app:deviations.

#### Compute
All model inference and training used a single laptop GPU (NVIDIA RTX 5070, 8 GB). Exact learners, SAT computations and all analyses ran on the CPU.

#### Use of AI assistance
Substantial AI assistance was used throughout this project. An AI system (Claude, Anthropic), working under the author's direction, contributed to the study design, wrote the code and the preregistrations, ran the experiments, analyses and audits, and drafted the text of this paper. The author takes responsibility for the content.

#### Competing interests
The author also wrote the compute–retrieve preprint tested in Experiment 5 [@ramdan2026cr]. Experiment 5 was preregistered before scoring, its hypothesis and decision rule were fixed in advance, and its result, which does not support the preprint's detector on this contrast, is reported in full. The author declares no other competing interests.

#### Code and data availability
Code, stimuli, results and preregistrations are available at `https://github.com/Somebodyhere101/compounding-fingerprint`. The repository contains the stimulus generator, the exact forcing-class labels, the idealised learners, the decision rule, all scoring and training scripts, every raw result file, the audit scripts and the preregistrations.
