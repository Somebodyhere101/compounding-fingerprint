## The compounding fingerprint {#sec:fingerprint}

The compounding fingerprint has four parts: a table-completion task in which earlier tables can be shown or withheld, an exact label for every queried cell, history conditions that isolate what the history contributes, and a preregistered decision rule that maps the resulting signature to a learner class.

### Task {#sec:task}

A trial presents one or more operation tables as facts `x * y = z`, one per line in random order, over words that are single tokens. Each table is preceded by a neutral header (`Table A:`, `Table B:`, …). The first t tables, with t ∈ {0, 1, 2, 4}, are the **history**. They are complete, and each uses a fresh set of words, so no word is reused across the tables of a trial. The last table is the **target**, which reveals k of its n² cells, the **cue**. Every unrevealed target cell is then queried independently by appending a newline and `wa * wb =`. The answer is a forced choice among the target's n words, scored by the model's next-token log-probability of each word with a leading space.

A single generator produced all stimuli. It assigns fresh words to each table through a random word–element bijection (only the leak control repeats a table, by design), randomises fact order, blacklists numerals, ordinals and identity-suggestive words, and draws from 101 words that are single tokens, with and without a leading space, in every tokenizer used. Because every table carries its own words, a base history can help with the target only through the structure of its tables.

### Forcing classes {#sec:classes}

Every queried cell is labelled exactly with the weakest axiom class under which the cue determines its value. This gives six **forcing classes**:

- **Latin-forced (L)**: every Latin square consistent with the cue agrees on the cell.
- **Loop-forced (O)**: every loop consistent with the cue agrees, so identity and cancellation suffice.
- **Associativity-forced / G-forced (G)**: every group table on the same labels agrees; the cell is forced only through associativity.
- **Ab** and **Cy**: the cell is forced only once commutativity or cyclicity is added.
- **U**: the cell is undetermined.

Latin classes were computed by enumeration for n ≤ 5 and with a SAT solver [@ignatiev2018pysat] otherwise, loop classes by enumeration for n ≤ 6 and SAT otherwise, and group classes by enumerating all labelled tables of every group of order n. The exact learners' code recomputes the classes; it agreed with the stimulus labels on all 57,200 queried cells of the main design (0 mismatches), and SAT agreed with exact counts on 913 trials.

Accuracy on G cells is the primary measure. A G cell is by definition not determined by identity and cancellation, so a learner that uses only Latin or loop constraints can do no better on it than guessing among the values a loop completion of the cue allows (the loop level); only a learner that uses group structure can exceed that level. Accuracy on L cells, $\mathrm{acc}_L$, measures the most basic inference, and chance on every class is 1/n (0.167 at n=6).

### Conditions {#sec:conditions}

The fingerprint compares G-cell accuracy across five history conditions (@tab:conditions), the base condition and four controls, each of which isolates one route by which a history could raise accuracy.

Table: The five history conditions. Each trial has t history tables followed by one partial target table. {#tab:conditions}

| condition | history | target | controls for |
|---|---|---|---|
| base | random groups of order n | random group | — |
| scramble | Latin squares with no two-sided identity, matched in fact count and token length | random group | context length |
| held-out types | n=6: Z6 only; n=8: abelian types only (Z8, Z4×Z2, Z2³) | an unseen type (n=6: S3; n=8: D4 or Q8) | exemplar matching |
| leak | the last history table is the target's own labelled table | random group | recall |
| loops (n=5) | group-like non-associative loops | Z5, or a loop | associativity |

The **scramble control** contains no group but matches a base history in facts and tokens (6 of the 400 n=6 scramble tables have a one-sided identity). Base minus scramble G accuracy at t=4, written t4 − scramble, therefore measures what group histories add beyond context length. With **held-out types**, no history table shares the target's isomorphism type, so an exemplar matcher has nothing to copy while an abstracting learner can still apply the axioms. The **leak control** makes every queried cell answerable by recall, a positive control for context use. The **loops** condition supplies histories that have an identity but are not associative, testing whether a learner infers "group" from an identity alone.

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

Ties are broken uniformly at random with a fixed seed. The rule distinguishes five learner classes: **R** (no inference), **N** (inference without use of the history), **N_iso** (exemplar reuse), **A** (abstraction) and **B-type** (the structural prior is in place without history). The preregistered G-cell predictions at n=6, k=10 differ sharply. R stays at chance (0.17) and N at 0.30 for every t. N_iso rises gradually (0.30, 0.65 and 0.96 at t = 0, 1 and 4), with held-out accuracy near 0.31. A steps from 0.30 at t=0 to 1.00 from t=1, with held-out accuracy 1.00. B and Bh stay at or near 1.00. Abstraction and exemplar matching therefore both compound to near ceiling by $t = 4$, so the compounding contrast alone cannot tell them apart; in the rule, held-out transfer does, and the shape statistic (@sec:rule) adds a secondary step-versus-gradual distinction.

An eighth learner, **C (count shortcut)**, was added *post hoc*, after the trained networks were found to use it (@sec:r-trained). C answers uniformly among the target words that occur least often as a result in the revealed cue. It uses no cell-level, history or axiom information.

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

Step 1 asks whether the model infers at all: below 1/n + 0.25 (0.417 at n=6), even cells that the Latin property alone determines are not solved, and the model is class R. Step 2 rescales accuracy so that 0 is chance and 1 is the model's own Latin accuracy. Step 3 identifies a learner whose group prior is already in place at t=0. Step 4 takes the higher of the t=0 and scramble values as baseline, so that a gain from context length alone does not count as compounding. Step 5 asks whether the gain survives when no history table shares the target's type.

The rule always also reports the abstraction fraction $w$ = held/comp and the shape statistic g(G t4) − g(G t1). A shape near 0 indicates a step, as for A; N_iso's gradual rise gives about 0.3. At n=8 the same quantities form a secondary analysis, with raw G accuracy reported alongside, because L cells are rare there (fewer than 40 per cell) and $\mathrm{acc}_L$ is noisy. For class R the rescaled quantities are not meaningful, since g is undefined when $\mathrm{acc}_L$ is near chance; class R means only that $\mathrm{acc}_L$ is below threshold, and recall is read separately from the leak control. Applied to the idealised learners, the rule returns each learner's own class at both n=6 and n=8, on both accuracy and mean $p_{\mathrm{gold}}$, with 0 misclassifications; Asl is classified A (@sec:r-ideal).
