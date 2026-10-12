## The compounding fingerprint {#sec:fingerprint}

The compounding fingerprint has four parts: a table-completion task, an exact label for every queried cell, history conditions that isolate what the history contributes, and a preregistered decision rule that maps the resulting signature to a learner class.

### Task {#sec:task}

A trial presents one or more operation tables as facts `x * y = z`, one per line in random order, over words that are single tokens. Each table is preceded by a neutral header (`Table A:`, `Table B:`, …). The first t tables, with t ∈ {0, 1, 2, 4}, are the **history**. They are complete, and each uses a fresh set of words, so no word is reused across the tables of a trial. The last table is the **target**, which reveals k of its n² cells, the **cue**. Every unrevealed target cell is then queried independently by appending a newline and `wa * wb =`. The answer is a forced choice among the target's n words, scored by the model's next-token log-probability of each word with a leading space.

A single generator produced all stimuli. It assigns fresh words to each table through a random word–element bijection (only the leak control repeats a table, by design), randomises fact order, blacklists numerals, ordinals and identity-suggestive words, and draws from 101 words that are single tokens, with and without a leading space, in every tokenizer used. Because every table carries its own words, a base history can help with the target only through the structure of its tables.

### Forcing classes {#sec:classes}

Every queried cell is labelled exactly with the weakest axiom class under which the cue determines its value. This gives six **forcing classes**:

- **Latin-forced (L)**: every Latin square consistent with the cue agrees on the cell.
- **Loop-forced (O)**: every loop consistent with the cue agrees, so identity and cancellation suffice.
- **Associativity-forced / G-forced (G)**: two loops consistent with the cue disagree on the cell, but every group table consistent with the cue agrees.
- **Ab** and **Cy**: the cell is forced only once commutativity or cyclicity is added.
- **U**: the cell is undetermined.

The classes are checked in this order, so each cell receives the weakest class that forces it. Every loop of order 4 or less is a group, so G cells first exist at order 5.

A worked example at $n = 5$ shows the difference. Label the elements e, a, b, c and d, and let the cue reveal the row and column of e (so e is the identity), a·a = b and a·b = c. Then a·c = d is Latin-forced: row a still lacks d and e, and column d already holds d. In a group, b·b = (a·a)·b = a·(a·b) = a·c = d. The loop in @tab:worked-loop also fits the cue and has a·c = d, but b·b = e: in it (a·a)·b = e while a·(a·b) = d, so associativity fails. Exhaustively, 6 of the 161,280 Latin squares of order 5 fit this cue, all 6 are loops, and between them b·b takes the values e, a and d; exactly one group table fits (Z5), with b·b = d, so b·b is G-forced (`paper/worked_example.py`). At $n = 5$ every commutative loop is the cyclic group, so commutativity alone would also force this cell; at $n = 6$ and $n = 8$, which have non-abelian groups, G cells exist that commutativity cannot explain.

Table: A loop of order 5 that fits the worked example's cue (rows: left factor; columns: right factor) but is not associative: b·b = e, whereas every group fitting the cue gives d. {#tab:worked-loop}

| · | e | a | b | c | d |
|---|---|---|---|---|---|
| e | e | a | b | c | d |
| a | a | b | c | d | e |
| b | b | d | e | a | c |
| c | c | e | d | b | a |
| d | d | c | a | e | b |

All classes were computed exactly, by enumeration or with a SAT solver [@ignatiev2018pysat], and an independent recomputation agreed with the stimulus labels on all 57,200 queried cells of the main design (@app:impl).

Accuracy on G cells is the primary measure. A G cell is by definition not determined by identity and cancellation, so a learner that uses only Latin or loop constraints can do no better on it than guessing among the values a loop completion of the cue allows (the loop level); only a learner that uses more than identity and cancellation can exceed that level. Exceeding it shows use of group structure beyond identity and cancellation, which need not be associativity: copying a revealed transposed cell (commutativity) or a remembered table also exceeds it, which is why @sec:r-ica splits G cells by whether the transpose is revealed. Accuracy on L cells, $\mathrm{acc}_L$, measures the most basic inference, and chance on every class is 1/n (0.167 at n=6).

### Conditions {#sec:conditions}

The fingerprint compares G-cell accuracy across five history conditions (@tab:conditions): the base condition and four controls, each isolating one route by which a history could raise accuracy.

Table: The five history conditions. Each trial has t history tables followed by one partial target table. {#tab:conditions}

| condition | history | target | controls for |
|---|---|---|---|
| base | random groups of order n | random group | — |
| scramble | Latin squares with no two-sided identity, matched in fact count and token length | random group | context length |
| held-out types | n=6: Z6 only; n=8: abelian types only (Z8, Z4×Z2, Z2³) | an unseen type (n=6: S3; n=8: D4 or Q8) | exemplar matching |
| leak | the last history table is the target's own labelled table | random group | recall |
| loops (n=5) | group-like non-associative loops | Z5, or a loop | associativity |

The **scramble control** contains no group but matches a base history in facts and tokens (6 of the 400 n=6 scramble tables have a one-sided identity), so base minus scramble G accuracy at t=4, written t4 − scramble, measures what group histories add beyond context length. With **held-out types**, no history table shares the target's isomorphism type, so an exemplar matcher has nothing to copy while an abstracting learner can still apply the axioms. The **leak control** makes every queried cell answerable by recall (a positive control for context use), and the **loops** condition tests whether a learner infers "group" from an identity alone.

At n=6 (k=10) and n=8 (k=16) the main design covers base t ∈ {0, 1, 4}, scramble t=4 and held-out t ∈ {1, 4}; n=6 alone adds base t=2 and leak t=1. At n=5 (k=6) it covers base t ∈ {0, 4} and the two loop cells at t=4. With 100 trials per cell, every idealised learner, pretrained model and trained network sees the same 1,800 trials and 57,200 scored cells (the In-Context Algebra model sees only the cells that fit its 16-symbol vocabulary, @sec:ica). The order n=6 is primary and n=8 secondary.

### Idealised learners {#sec:learners}

To check that the fingerprint separates the hypotheses it targets, we scored seven exact algorithms on the identical trials:

- **R** recalls only.
- **N** chooses uniformly among the Latin-feasible values and otherwise ignores the history; like every other preregistered learner, it first copies a cell that appears verbatim in the history (R does only this), which is possible only in the leak control.
- **N_iso** matches the cue against the history tables up to relabelling (exemplar matching), with a Latin fallback.
- **A** induces axioms (identity, associativity, commutativity, cyclicity) from the history tables, then applies them.
- **Asl** is like A, but infers "group" from an identity alone.
- **B** is Bayes-optimal under the true environment prior.
- **Bh** is hierarchical Bayes over the Latin, loop, group, abelian and cyclic environments.

Ties are broken uniformly at random with a fixed seed. The rule distinguishes five learner classes: **R** (no inference), **N** (inference without use of the history), **N_iso** (exemplar reuse), **A** (abstraction) and **B-type** (the structural prior is in place without history). In the preregistered G-cell predictions at n=6, k=10, R stays at chance (0.17) and N at 0.30 for every t; N_iso rises gradually (0.30, 0.65 and 0.96 at t = 0, 1 and 4), with held-out accuracy near 0.31; A steps from 0.30 at t=0 to 1.00 from t=1, with held-out accuracy 1.00; and B and Bh stay at or near 1.00. Abstraction and exemplar matching thus both compound to near ceiling, and only held-out transfer, with the shape statistic (@sec:rule) as a secondary step-versus-gradual distinction, tells them apart.

An eighth learner, **C (count shortcut)**, was added *post hoc*, after the trained networks were found to use it (@sec:r-trained). C answers uniformly among the target words that occur least often as a result in the revealed cue, using no cell-level, history or axiom information.

### Decision rule {#sec:rule}

The decision rule (@tab:rule) was *preregistered* and frozen before any model output. It is applied per model to the n=6 cells, on accuracy and, separately, on mean $p_{\mathrm{gold}}$, the probability of the correct word under a softmax over the n candidates.

Table: The preregistered decision rule, applied in order; the first step whose criterion holds assigns the class. G t0, G t4, G scr4 and G held4 are G-cell accuracies at base t=0, base t=4, scramble t=4 and held-out t=4. {#tab:rule}

| step | quantity | criterion | class |
|---|---|---|---|
| 1 | $\mathrm{acc}_L$, L-cell accuracy at base t=0 | $\mathrm{acc}_L < 1/n + 0.25$ | R |
| 2 | rescaling $g(x) = (x - 1/n)/(\mathrm{acc}_L - 1/n)$ | — | — |
| 3 | g(G t0) | $\geq 0.85$ | B-type |
| 4 | base = max(g(G t0), g(G scr4)); comp = g(G t4) − base | comp < 0.15 | N |
| 5 | held = g(G held4) − base | held ≥ 0.6·comp | A; otherwise N_iso |

Step 1 asks whether the model infers at all: below 1/n + 0.25 (0.417 at n=6), even cells that the Latin property alone determines are not solved. Step 2 rescales accuracy so that 0 is chance and 1 is the model's own Latin accuracy; step 3 identifies a group prior already in place at t=0; step 4 takes the higher of the t=0 and scramble values as baseline, so that a gain from context length alone does not count as compounding; and step 5 asks whether the gain survives when no history table shares the target's type.

The rule always also reports the abstraction fraction $w$ = held/comp and the shape statistic g(G t4) − g(G t1). A shape near 0 indicates a step, as for A; N_iso's gradual rise gives about 0.3. At n=8 the same quantities form a secondary analysis, with raw G accuracy reported alongside, because L cells are rare there (fewer than 40 per cell) and $\mathrm{acc}_L$ is noisy. For class R the rescaled quantities are not meaningful, since g is undefined when $\mathrm{acc}_L$ is near chance; class R means only that $\mathrm{acc}_L$ is below threshold, and recall is read separately from the leak control.
