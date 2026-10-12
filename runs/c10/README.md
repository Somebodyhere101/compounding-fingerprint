# Design phase (2026-10-06)

This folder holds the design phase of the study: the code that generates the stimuli and labels every queried cell, and the report and notebook from which the Experiment 1 preregistration was built.

- `code/` holds:
  - the stimulus generator (`stimuli.py`) and the exact forcing-class engine (`engine.py`);
  - group, loop and Latin tables;
  - the idealised-learner simulations and the minimum-defining-set computations.

  `exp1/`, `exp2/`, `exp3/tests.py` and `exp6/loop_level.py` import from it.
- `out/` holds the outputs and logs of those computations. `report_tables_partial.md` and `sim_n4_test.json` are intermediate test outputs that nothing references; they are kept for completeness.
- `report.md` and `log.md` are the design report and running notebook. An AI research agent (Claude) wrote them during a time-boxed design run. Read them with these caveats:
  - "Candidate 10", "the brief" and "other candidates" refer to the author's initial task description and to parallel design attempts, none of which are included.
  - The audits did not cover the minimum-defining-set results or the claims marked **[PROVED]** or "most likely to be new". None of these are part of the paper.
  - `report.md` says the order-10 computation was still running when it was written. It finished afterwards; the result is in `out/md_n10.json` (see `log.md`).
  - Remarks about the agent's own tooling, for example that a report file could not be written, are left as they were.
