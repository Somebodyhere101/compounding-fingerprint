# Positioning: what is new, and where we could be scooped

*Pre-manuscript draft, kept as provenance. It was only partly verified, and its minimum-defining-set claims come from the unaudited design report (`../runs/c10/`) and are not part of the paper. The manuscript in `manuscript/` is authoritative.*

*Draft, 2026-10-07. Bracketed numbers refer to the bibliography in `related_work.md`.*

## What this project adds

1. **An exact forcing-class fingerprint.** Every queried cell is labelled exactly, by enumeration or SAT, with the weakest axiom class that forces its value:
   - **L**: Latin-forced;
   - **O**: forced by identity and cancellation (loop axioms);
   - **G**: forced only by associativity;
   - **Ab / Cy**: forced by commutativity / cyclicity;
   - **U**: undetermined.

   In-Context Algebra [1] attributes accuracy to mechanisms through targeted data distributions. We instead score each cell by the axiom that makes it inferable. G-cells isolate associativity, which [1] found to be the weakest route (60.2%). G-cells cannot exist at n ≤ 4, because every loop of order ≤ 4 is a group.
2. **Exact idealised baselines on identical trials.** R, Np/N, N_iso, A, Asl, B and Bh are run as exact algorithms on the same stimuli the models see. Two results follow:
   - At prime n, N_iso and A are provably identical. Exemplar matching and abstraction can only be told apart on held-out isomorphism types, at composite n.
   - The exemplar learner's compounding curve equals the Good–Turing missing mass of the environment's type distribution.

   Prior memorising-vs-generalising work [6, 7] fits Bayesian predictors to *pretraining* task sets. Here the "seen tasks" arrive in the prompt under fresh labels.
3. **A preregistered, lapse-normalised decision rule** (R / N / N_iso / A / B-type). Its thresholds were frozen before any model output was seen. In simulation it classifies idealised learners correctly in ≥97% of studies at 100 trials per cell.
4. **Pretrained-LLM scaling with controls.** The conditions are length-matched scramble histories, held-out types, a leak positive control and a loop condition. Experiment 1 found every model from 0.6B to 1.7B to be R, with no compounding. Experiment 2 extends the series to Qwen3-4B/8B-Base and OLMo-2-7B (nf4) and includes a quantisation check. We found no published evaluation of pretrained LLMs on in-context group-table completion with fresh labels (see the appendix).
5. **Controlled from-scratch transformers.** Small models trained on the same stimulus distribution, where in-weights and in-context priors can be manipulated directly. This is the bridge to [1] and to [8]'s in-weights-versus-in-context contrast. *(Planned; its status should be confirmed by the lead.)*
6. **Minimum defining sets of group tables.** These are exact values for all groups of order ≤ 9. The formula ⌈(n−1)/2⌉ holds for orders 2–7 and fails from order 8: Z4×Z2, D4, Q8 and Z8 need 5, 5, 5 and 4 cells, Z2³ needs 6, and Z9 and Z3×Z3 need 5. We also prove a lower bound ⌈(n−1)/3⌉, which is tight at Q8 under the relabelling-only prior. This is the group-class analogue of Latin critical sets and teaching dimension [23].

## Three strongest "already done" risks

**Risk 1. In-Context Algebra [1], plus any follow-up from the same lab. Highest risk.** It already has fresh per-sequence symbol assignments, several groups per sequence on disjoint token sets, held-out order-8 groups, and mechanisms that line up with our O-cells.
- *Exposure:* a reviewer may say that our from-scratch arm repeats [1], and that "does a second group in context help?" is a small extension of their multi-group sequences.
- *Defence:* [1] measures no dependence on the number of history tables, uses no exact forcing classes, has no exemplar baseline at equal budget, and evaluates no pretrained LLMs.
- *Action:* cite [1] prominently and frame our from-scratch arm as complementary. Our fact format differs (full tables plus a partial target, not interleaved facts). Before submission, check for a 2026 follow-up from the same lab (none found today; Semantic Scholar lists 3 citing papers, none on this task).

**Risk 2. Matsutani et al. 2026 [20].** They report that demonstrations from *other worlds*, including other topologies, improve in-context world modelling in pretrained Qwen3 4–14B base models, with gains that saturate with shots. That is compounding across relabelled worlds in a graph domain.
- *Exposure:* "cross-episode in-context improvement under relabelling is already shown."
- *Defence:*
  - their worlds reuse one vocabulary;
  - we found no length-matched scramble control (Experiment 1 shows such controls matter: the one significant raw t-effect vanished against scramble);
  - there are no idealised baselines and no exemplar-vs-abstraction test;
  - accuracy is not split by what makes an answer inferable.
- Related: Park et al. [18] (in-context representation reorganisation, Llama-3.1-8B) and Lepori et al. [19] (such representations are largely inert; about 40% on a 4×4 grid for open models).

**Risk 3. The combinatorics of defining sets: Balister–Illingworth–Scott–Stein–Tamitegama [24]; Hatami–Qian [23]; Huh [4, 5].**
- *Exposure:* [24] studies how many table queries reconstruct a group with unknown element order. Their "n−1 elements seen" threshold is the same quantity as our lower-bound lemma (any defining cue must mention n−1 labels). A forthcoming preprint could contain fixed-cue minima or the ⌈(n−1)/3⌉ bound.
- [23] already identifies critical sets with teaching dimension, so the framing is not new; only the class (group tables) and the values are.
- [4, 5] own the phrase "Cayley-table completion" for a different (factorisation) problem.
- *Action:*
  - present md as a computed, exhaustively verified table plus a short lemma;
  - cite [24] as concurrent, from the talk abstract only;
  - re-search arXiv for their preprint right before submission (none found 2026-10-07);
  - keep the novelty claim at "values for n = 8–9 and the bound".

A lesser risk is the conceptual one: the N_iso vs A distinction is the memorising-vs-generalising dichotomy of [6, 7], and the exemplar-vs-rule contrast of [8]. Claim the in-context, fresh-label version and the missing-mass identity, not the dichotomy itself.

---

## Appendix: calibration for Experiment 2 (task 3)

**Search result.** We searched for 2025–2026 work that evaluates *pretrained* LLMs on in-context completion of Cayley/group tables, or on permuted-symbol algebra with fresh labels. We found none. Methods:
- WebSearch, about 12 queries: Cayley table / group operation / quasigroup / Latin square / permuted modular arithmetic × LLM / in-context / pretrained / 2025–2026;
- arXiv API phrase queries;
- Semantic Scholar citations of [1].

The closest calibrators are below. Scale is the variable that matters.

| work | models | finding relevant to Exp 2 |
|---|---|---|
| Todd et al. [1] | 4-layer transformer trained from scratch | Near-perfect on fresh-symbol group facts; associative composition 60.2%. No pretrained LLMs. |
| Matsutani et al. [20] | Qwen3-4B/8B/14B-Base (+Llama-3, Gemma, Ministral) | Multi-world demonstrations help on a graph task already at 4B base; gains saturate with shots. |
| Lepori et al. [19] | Gemma-3 4/12/27B-it, OLMo-2-13B; GPT-5, Gemini-2.5 | Open models ≈40% on 4×4-grid adaptive world modelling; frontier reasoners degrade on 2-D structure. |
| Park et al. [18] | Llama-3.1-8B (also Llama-3.2-1B, Gemma-2-2B/9B) | Representations reorganise to an in-context graph once context is long enough; transition context grows with graph size. |
| Wei et al. [14] | GPT-3, PaLM, Flan-PaLM families | Learning from semantically unrelated labels emerges mainly at the largest scales. |
| Wu et al. [15]; Gupta et al. [16] | GPT-4-class; several LLMs | Strong degradation on counterfactual arithmetic; LLMs below n-gram baselines on random languages. |

**Implication.**
- Nothing in the literature predicts G-cell (associativity) inference at 4–8B from a raw table prompt. The nearest positive result [20] shows cross-world gains at 4B+ base models on a simpler graph task, but without the controls we use.
- A plausible Experiment 2 outcome is Latin-level (Np) or still R at 4–8B, with any compounding confined to O-cells. A null at 8B would match [19].
- This makes the from-scratch arm, and possibly a larger model run off this GPU, the likely place to observe A vs N_iso.
- These are expectations, not results; Experiment 2 outputs were not analysed here.
