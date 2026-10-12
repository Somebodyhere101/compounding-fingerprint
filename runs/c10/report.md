# Candidate 10: a compounding fingerprint for transformers. Report

All paths are relative to `runs/c10/`. Python: `C:\Users\%USERNAME%\AppData\Local\Temp\ccv\Scripts\python.exe`. Every command below is run from `runs/c10/code/`. Raw outputs are in `out/`, and the running notebook is `log.md`.

## 1. Summary

**Result most likely to be new.** Take the minimum cue (minimum defining set) that pins down one labelled Cayley table among *all group tables* on the same labels. It equals ⌈(n−1)/2⌉ for every group of order 2–7, which confirms the brief's data. The formula fails from order 8 onward: Z8 4, Z4×Z2 5, Z2³ 6, D4 5, Q8 5, Z9 5, Z3×Z3 5 **[VERIFIED n≤9]**, checked by two independent exhaustive methods. A general lower bound ⌈(n−1)/3⌉ for n≥5 is **[PROVED]**, and Q8 attains it under the relabelling-only prior. Confidence: (a) correct 95%, (b) new 55%.

**Main design findings for the brief's question:**
- A Bayes learner that knows the true prior cannot compound. This holds by definition **[PROVED]**.
- A hierarchical Bayesian with a balanced hyperprior moves k* by at most 1 cell (n = 4–7). Bounded learners move it by 7 to 28 cells **[HEURISTIC: Monte Carlo]**.
- At prime n, a notebook that does isomorphism matching is *identical* to the abstraction learner **[PROVED]**, so the Q1 curves cannot tell them apart. Held-out isomorphism classes can **[PROVED + MC]**. On group-forced cells the abstraction learner scores 1.00, against 0.31 for the exemplar notebook at n=6 and 0.22 at n=8.
- The exemplar notebook's compounding curve is the environment's missing mass, Σ_H p_H(1−p_H)^t **[PROVED identity; approximation checked by MC]**.
- A lapse-corrected decision rule classifies R/N/N_iso/A/B correctly in ≥97% of simulated studies at 100 trials per design cell **[HEURISTIC]**.

## 2. Clean problem statement, with repairs

**Objects.** A labelled table T is an n×n array on n word-labels with T[a,b] = a∗b. An environment E is a distribution over isomorphism types plus a uniformly random labelling with fresh words. The default is iso-uniform: each type has equal probability. In a trial:
- the context holds t complete history tables T_1..T_t, on pairwise disjoint vocabularies;
- it then holds a target T, also on a fresh vocabulary, with a uniformly random set S of k revealed cells (the cue);
- every unseen cell is queried, and scoring is forced choice among the n target words.

**Hypothesis classes.** Latin ⊃ loop (Latin square with a two-sided identity) ⊃ group ⊃ abelian ⊃ cyclic. Each class is closed under relabelling. A cell c is *X-forced* by cue K if every table of class X that agrees with K has the same value at c.

**Repair 1: split "group-forced" into two classes.** Each unseen cell gets one label:
- L: Latin-forced;
- O: loop-forced but not L;
- G: group-forced but not O, so associativity is genuinely needed;
- Ab or Cy: abelian- or cyclic-forced, but not group-forced;
- U: undetermined.

The brief's "group-forced-but-not-Latin" class is O ∪ G. More than half of it (O) is forced by identity plus cancellation alone, the mechanisms In-Context Algebra already found in small transformers. Every loop of order ≤4 is a group, so **G is empty for n≤4 and associativity cannot be tested at n≤4.** The repair makes the question sharper. It makes n=4 useless for associativity.

**Repair 2: two different notions of k\*.** The brief's "exact data" k* is a teaching-set quantity: md_X(T), the minimum size of a cue that leaves T as the only table of class X consistent with it. The cue sizes at which a model "snaps" with *random* cues are much larger. I define k*_θ(learner) = min{k : E[accuracy on unseen cells] ≥ θ} with θ = 0.9, averaged over the environment's types and uniform cue positions.
- n=6, group prior: md = 3, but k*_0.9 = 7.
- n=6, Latin: the known smallest critical set of a back-circulant square is ⌊n²/4⌋ = 9 (literature), but k*_0.9 = 20.

Both quantities are reported.

**Repair 3: name the real competitor of A.** Under fresh labels, retrieval by surface match finds nothing by construction. "N" is therefore t-flat by definition and is not a useful competitor of A. The barrier the brief names (retrieval masquerading as abstraction) is a notebook whose inference budget suffices for **isomorphism matching**. I call it **N_iso**: its hypotheses are all relabellings of the stored tables' types, and it falls back to Latin inference when none fits.

**Repair 4: Bayes benchmarks and environment weighting.**
- (B) with the true prior is t-flat by definition (Prop C). I add **B_hier**: Bayes over environments {Latin, loop, group, abelian, cyclic}, each available where its likelihood can be computed exactly.
- The environment weighting must be pre-registered. Iso-uniform and label-uniform weighting change B's MAP answers. Example: n=6, Z6 targets, iso-uniform. B's accuracy on abelian-forced cells is 0.08, because S3 relabellings are three times heavier (`sim_n6.json`, base, B, Ab cells).

**Learners, implemented as exact algorithms (`code/sim.py`).**
- **R**: verbatim lookup only; chance (1/n) on unseen cells under fresh labels.
- **Np**: Latin propagation (naked and hidden singles) to a fixpoint. This is the bounded notebook.
- **N**: full Latin feasibility, by enumeration (n≤5) or SAT (n≥6).
- **Nb**: Latin MAP, n≤5.
- **N_iso / Nisob**: exemplar isomorphism matching, possibilistic / MAP.
- **A / Ab**: induces the axioms (identity, associativity, commutativity, cyclicity) that hold in every history table, then completes within the class those axioms define. It falls back cyclic→abelian→group→loop→Latin when the cue contradicts the class.
- **A_rec**: the same as A, using only the last two tables.
- **Asl**: "sloppy" A. If every history table has an identity, it assumes groups without checking associativity.
- **A_w**: group axioms fixed in weights.
- **B, Bposs**: the true prior.
- **Bh, Bh_w (hyperprior 2¹⁰:1 for groups), Bh_cp (last two tables)**.

Possibilistic learners guess uniformly among feasible values. MAP learners use exact counts.

**Equal resources.** A and N_iso enumerate at most G(n) candidate tables: 480 at n=6, 840 at n=7, 22080 at n=8. N (full Latin) has the *larger* budget, up to L(n). Np is polynomial, yet it has the same k* as N in every computed case (T1). So A vs N_iso is a comparison at equal budget, and A vs Np gives the notebook more structure, not less.

## 3. Corrections to the BACKGROUND

1. **"Under the group prior the minimum cue is ⌈(n−1)/2⌉ for every group of order 2–7."** Correct as stated, including V4 (2) and S3 (3) **[VERIFIED n≤7]**. It does **not** extend.
   - n=8: Z8 4, Z4×Z2 5, Z2³ 6, D4 5, Q8 5, against a formula value of 4.
   - n=9: Z9 5, Z3×Z3 5, against a formula value of 4.

   **[REFUTED: Z4×Z2 needs 5 > ⌈7/2⌉]**. From order 8 on, md depends on the isomorphism type, not only on n. Any n=8 held-out design must calibrate cue sizes per type.
2. **"Recall, Latin and composing learners predict different fingerprints."** True for R, Np and N against A. False for the notebook that matters: at n = 5 and 7, N_iso and A have identical hypothesis sets for every t ≥ 1, so they give identical fingerprints (Prop B2).
3. **Literature items.** All verified by opening the sources (§7).
   - The closest prior work is *In-Context Algebra* (Todd, Brinkmann, Gandikota, Bau; arXiv 2512.16902, ICLR 2026). It trains small 4-layer transformers on "x y = z" facts with a random variable↔element map per sequence; training covers cyclic groups C3–C10 and dihedral D3–D5, and all order-8 groups are held out. They report commutative copying, identity recognition and closure-based cancellation, with associative composition weakest (60.2%).
   - It tests no pretrained Qwen/Llama models and no cross-group compounding (my reading of the HTML version).
   - Their mechanisms map exactly onto the O-cells; associativity maps onto the G-cells.
4. **Barrier.** The three named confounds (retrieval as abstraction, token leakage, context length) are real. Four more need control:
   - environment weighting (Repair 4);
   - the cell-class mix depends steeply on k: at the useful cue sizes almost no cell is Latin-forced (n=6, k=10: 1.3 L-cells vs 11.8 G-cells per trial);
   - n≤4 cannot test associativity;
   - model lapses, which need normalising by the model's own accuracy on Latin-forced cells (§5.5).

## 4. Computations

Notation:
- **MC** means exact computation per cue, Monte Carlo over cue positions. The number of cues per (target type, k) is M = 400 (n=4), 80 (n=5), 150 (n=6, 7), 20 (n=8, k step 4).
- Accuracy standard errors are ≤ 0.02. k* values are uncertain by about ±1 cell (±4 at n=8).
- An exhaustive check over all 1820 cue sets per target at n=4, k=4 (`python exact_small.py 4 4`) matches the MC values to within 0.02. Examples: N on O-cells 0.4289; N_iso on O-cells for t=0,1,2: 0.4289, 0.8015, 0.9008; B on all cells 0.9544.

**4.1 Counts.** `python tables.py` gives:
- labelled groups n=1..8: 1, 2, 3, 16, 30, 480, 840, 22080 (= OEIS A034383);
- Latin squares n≤5: 1, 2, 12, 576, 161280 (= A002860);
- labelled loops n≤6: 1, 2, 3, 16, 280, 56448.

`python md_extra.py n10` enumerates the 1088640 labelled groups of order 10 as Aut-coset representatives, which equals A034383(10). Non-associative loops of order 5 form 5 isomorphism classes, of sizes 40, 40, 10, 40, 120 (`python test_engine.py`). SAT feasibility agrees with enumeration on every cue/class tested (0 mismatches over 320 checks at n=4–6).

**4.2 Minimum defining sets [VERIFIED n≤9].**

Commands:
- `python defsets.py 2,3,4,5,6,7 ../out/defsets.json`
- `python defsets.py 8 ../out/defsets8.json` (bitset DFS over all cue subsets)
- `python defsets_ilp.py 2,3,4,5,6,7,8 group ../out/md_ilp_2to8.json` and `python defsets_ilp.py 9 group ../out/md_ilp_9.json` (independent: minimum hitting set over all competing tables, CP-SAT, optimality certified)
- `python check9.py` (DFS: no defining cue of size ≤4 at n=9)
- `python md_extra.py iso` (relabelling-only prior)

| n | type | group prior | abelian prior | cyclic prior | loop prior | Latin prior | relabelling only | ⌈(n−1)/2⌉ |
|---|---|---|---|---|---|---|---|---|
| 2 | Z2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 3 | Z3 | 1 | 1 | 1 | 1 | 2 | 1 | 1 |
| 4 | Z4 | 2 | 2 | 2 | 2 | 4 | 2 | 2 |
| 4 | V4 | 2 | 2 | – | 2 | 5 | 1 | 2 |
| 5 | Z5 | 2 | 2 | 2 | 3 | 6 | 2 | 2 |
| 6 | Z6 | 3 | 3 | 3 | 5 | – | 3 | 3 |
| 6 | S3 | 3 | – | – | 6 | – | 3 | 3 |
| 7 | Z7 | 3 | 3 | 3 | – | – | 3 | 3 |
| 8 | Z8 | 4 | 4 | 4 | – | – | 4 | 4 |
| 8 | Z4×Z2 | **5** | 4 | – | – | – | 4 | 4 |
| 8 | Z2³ | **6** | 4 | – | – | – | 4 | 4 |
| 8 | D4 | **5** | – | – | – | – | 4 | 4 |
| 8 | Q8 | **5** | – | – | – | – | **3** | 4 |
| 9 | Z9 | **5** | 5 | – | – | – | **5** | 4 |
| 9 | Z3×Z3 | **5** | 5 | – | – | – | 4 | 4 |

The Latin column for n≤5 reproduces known smallest-critical-set values: back-circulant ⌊n²/4⌋ = 2, 4, 6, and 5 for the Klein square. Example minimum cues (labels 0..n−1, 0 = identity), each checked to leave exactly one group table:
- Z5: {1∗1=2, 2∗3=0};
- Z2³: {1∗1=0, 2∗2=0, 3∗4=7, 4∗5=1, 5∗6=3, 6∗7=1};
- Q8, relabelling only: {1∗5=4, 3∗6=4, 7∗2=5}.

Order 10 (Z10, D5) was still running at the time of writing; see §9.

**4.3 Evidence per history table [VERIFIED n≤6; n=7 uses L(7) from OEIS A002860].** `python evidence.py a`. λ = log₂ P_E(T)/P_E′(T) for one complete table, with E = iso-uniform groups:

| n | type | λ vs Latin (bits) | λ vs loops (bits) | log₂ L(n) |
|---|---|---|---|---|
| 4 | Z4 / V4 | 4.58 / 6.17 | −0.58 / 1.00 | 9.17 |
| 5 | Z5 | 12.39 | 3.22 | 17.30 |
| 6 | Z6 / S3 | 20.11 / 21.69 | 6.29 / 7.88 | 29.60 |
| 7 | Z7 | 36.09 | 17.11 | 45.81 |

The cue's own evidence at n=5 is e(k) = E log₂ P_grp(K)/P_Latin(K): k=2: 0.77, k=4: 3.91, k=6: 7.26, k=7: 8.49, k=10: 11.63, rising to 12.39. Against loops it saturates at 3.22 (`python evidence.py b 5 60`, MC).

**4.4 Baseline k\*_0.9(t), t = 0..4 [HEURISTIC: MC].** History is drawn from E, averaged over the environment. Command: `python sim.py n M kstep` for each n, then `python report_tables.py 0.9 > ../out/report_tables.md` (table T1).

| n | Np = N | N_iso | A | Asl | Bposs | B | Bh |
|---|---|---|---|---|---|---|---|
| 4 | 8 flat | 8,7,6,5,5 | 8,5,5,5,5 | 8,4,4,4,4 | 4 | 2 | 2,3,3,3,3 |
| 5 | 11 flat | 11,4,4,4,4 | 11,4,4,4,4 | 11,4,… | 4 | 4 | 4 flat |
| 6 | 20 flat | 20,17,15,10,8 | 20,7,7,7,7 | 20,7,… | 7 | 6 | 7,6,6,6,6 |
| 7 | 25 flat | 25,6,6,6,6 | 25,6,6,6,6 | 25,6,… | 6 | 6 | n/a |
| 8 (k step 4) | 40 flat | 40,40,40,36,36 | 40,12,12,12,12 | 40,12,… | 12 | 8 | n/a |

R never reaches 0.9.

**4.5 Composition of unseen cells at the design cue size** (table T2; mean cells per trial):

| n | k | L | O | G | Ab | U |
|---|---|---|---|---|---|---|
| 5 | 6 | 0.94 | 11.79 | 6.12 | 0 | 0.15 |
| 6 | 10 | 1.30 | 12.04 | 11.77 | 0.09 | 0.80 |
| 7 | 12 | 0.36 | 11.88 | 24.64 | 0 | 0.12 |
| 8 | 16 | 0.23 | 11.96 | 34.69 | 0.32 | 0.80 |

**4.6 Accuracy on G-cells at the design k, t = 0,1,2,3,4 [HEURISTIC: MC]** (table T3):
- n=5, k=6: N .33 flat; Nb .40 flat; N_iso = A = .33, 1.00, 1.00, 1.00, 1.00; B 1.00; Bh 1.00.
- n=6, k=10: N .30 flat; N_iso .30, .65, .83, .91, .96; A .30, 1.00, 1.00, 1.00, 1.00; B 1.00; Bh 1.00.
- n=8, k=16: N .22; N_iso .22, .38, .51, .61, .69; A .22, 1.00, 1.00, 1.00, 1.00.
- Loop learner (identity plus cancellation, no associativity; `python loop_learner.py 6 10 100`, `… 5 6 200`, `… 7 12 60`): O-cells 1.00; G-cells .42 (n=6), .46 (n=5), .34 (n=7).

**4.7 Held-out isomorphism class [HEURISTIC: MC]** (T4). The history contains only the "seen" types; the target is a held-out type.
- n=6, Z6-only history, S3 target, k=10, t = 1..4. G-cells: N_iso .31 (≈ Latin .30), A 1.00. k*: N_iso 21 flat, A 8.
- n=8, abelian-only history, D4 / Q8 targets, k=16. G-cells: N_iso .22 / .22, A 1.00 / 1.00.
- n=4, Z4-only history, V4 target, k=4, exhaustive. O-cells: N_iso .7076, A 1.0000.

**4.8 Missing-mass law** (`python coverage.py`). The prediction is acc_G(N_iso,t) = c_t + (1−c_t)·a₀, with c_t = 1−(1−1/m)^t.

| setting | source | t=0 | t=1 | t=2 | t=3 | t=4 |
|---|---|---|---|---|---|---|
| n=6 (m=2) | simulated | .298 | .653 | .827 | .913 | .957 |
| n=6 (m=2) | law | .298 | .649 | .825 | .912 | .956 |
| n=8 (m=5) | simulated | .217 | .380 | .508 | .609 | .688 |
| n=8 (m=5) | law | .217 | .373 | .499 | .599 | .679 |

At n=4, k=4 the law underestimates (.715 vs .802 at t=1), because there the cue of an unseen type is often consistent with the seen type (q ≈ 0.49).

**4.9 Wrong abstraction, n=5** (T5; history = t non-associative loops, uniform over the 5 isomorphism classes; Z5 target, k=6, t = 0..4):
- A: G .33 → .46, O .35 → 1.00, k* 11 → 8.
- Asl: G → 1.00.
- Bh: G 1.00 → .49 (**negative transfer**), O 1.00.
- N_iso: G .33 → .23, O .35 → .54.
- N: flat.
- B: 1.00 flat.

Loop targets after group history (A misapplies the group prior): all-cell accuracy at k=6 is A .82 vs Bposs(loop) .84. This is small, because 6-cell loop cues rarely embed in Z5.

**4.10 Lesion, (i)** (T6; history t=4; P1 = probability the learner's own completion of the stage-1 target is fully correct; stage-2 G-accuracy = P1·acc(t=1) + (1−P1)·acc(t=0)):

| setting | N | N_iso | A | B |
|---|---|---|---|---|
| n=6, k1=10: P1 | .00 | .87 | .93 | .95 |
| n=6, k1=10: stage-2 G-accuracy | .30 | .61 | .95 | 1.00 |
| n=5, k1=6: P1 | .00 | .99 | .99 | — |
| n=5, k1=6: stage-2 G-accuracy | .33 | .99 | .99 | — |

**4.11 A→B→C, (iii)** (n=6; 2 tables from {Z6, S3}, then 2 Z6 tables; `python abc_inweights.py abc 40`):
- S3 target, k*_0.9: N 24, N_iso 16, A_cum 8, A_rec 8, Bh 6, Bh_cp 8.
- Group-like loop target (87.5% of triples associative), k*_0.9: N 18, N_iso 18, A 14, Bh 14.
- Commutative-copy probe (`python ccopy.py 150`). Probe cells are (a,b) with (b,a) revealed and T(a,b) ≠ T(b,a); the copy-error rate is P(answer = T(b,a)). At k=3 / k=4:
  - abc history: N .16/.16, N_iso .47/.37, A_cum .47/.36, **A_rec .84/.61**, Bh .52/.29, **Bh_rec .84/.61**;
  - abelian-only history: A, Bh and N_iso all ≈ .84/.61–.66.

**4.12 Prior in weights, (iv)** (n=5; `python abc_inweights.py w 40`):

| target | history | A_w (rigid) | Bh_w (revisable, 2¹⁰:1 for groups) |
|---|---|---|---|
| Z5 | groups | k* 4 for all t | k* 4 |
| Z5 | loops | k* 4 for all t | k* 4 → 7 at t ≥ 1; accuracy at k=6: 1.00 → .82 |
| loops | either | k* 8 | k* 8 |

**4.13 Bayesian compounding clock [HEURISTIC: MC]** (`python evidence.py c 6 40`, `python evidence.py cL 5 60`). k*(t=0,1,2) for log₂ prior odds against the group-type environments:

| log₂ odds | n=6, loop competitor, Z6 | n=6, loop competitor, S3 | n=5, Latin competitor only |
|---|---|---|---|
| 0 | 7,7,7 | 7,7,6 | 4,4,4 |
| 5 | 9,7,7 | 9,7,6 | 4,4,4 |
| 10 | 12,8,7 | 9,7,6 | 7,4,4 |
| 20 | 12,12,10 | 9,9,9 | 11,6,4 |
| 40 | 13,12,12 | 12,9,9 | 11,11,11 |

**4.14 Power of the decision rule [HEURISTIC]** (`python power.py > ../out/power.txt`). Synthetic models of each type answer with a lapse rate (0, .2, .4) and within-trial correlation ρ (.3, .6). With the lapse-corrected rule (§5.5):

| trials per design cell | mean correct classification |
|---|---|
| 25 | ≥ .88 |
| 50 | ≥ .94 |
| 100 | ≥ .97 |
| 200 | ≥ .99 |

The uncorrected rule fails for B under 20% lapses: B is called N in about 70% of runs (first run, `log.md`).

## 5. Lines of attack

### 5.1 Line 1: what can compound under fresh labels (Bayesian learners)

**Prop C (no compounding when E is known) [PROVED].** History tables are independent of the target given E, and fresh vocabularies make the labellings independent. So P(T | K, H, E) = P(T | K, E). Any learner that predicts with P(·|K, E) has the same accuracy for every t, k and cell class. This is true by definition. Its use is to show that the brief's B is a ceiling, not a compounding learner.

**Prop D (Bayesian compounding is bounded by the prior weight) [PROVED; the general fact is KNOWN, the standard Bayes-mixture bound].** Let B_hier have hyperprior π with π(E) > 0. For every complete target T, reveal its cells one by one in any order. The total excess log-loss of B_hier at t=0 over the true-prior B is

  log₂ P_E(T) − log₂ P_mix(T) ≤ log₂(1/π(E)),

because P_mix(T) ≥ π(E)·P_E(T). History can only bring B_hier closer to B. So, summed over a whole target table, the log-loss benefit of *any* history is at most log₂(1/π(E)) bits: 2.32 bits for a uniform 5-environment hyperprior.

A learner whose t=0 hypothesis space is Latin stands to gain λ bits per table (§4.3: 12.4 at n=5, 20–22 at n=6, 36 at n=7). The ratio grows super-exponentially with n. In k* terms, the uniform-hyperprior Bh shifts by ≤1 cell (§4.4: 2→3 at n=4, flat at n=5, 7→6 at n=6). Np, N_iso and A shift by 3 to 28 cells **[HEURISTIC: MC]**.

**Prop E (compounding clock) [PROVED as an identity].** With two environments E, E′, hyperprior odds O = π(E′)/π(E), and t complete history tables from E, the posterior odds are O·2^(−Σᵢ λ(Tᵢ)). The learner at history t is exactly the t=0 learner with odds O_t = O·2^(−tλ̄) (λ is constant per type). Approximately, its snap point satisfies e(k*) ≈ log₂O − tλ̄, falling back to E′'s k* when e(k) can never reach that value.

Check at n=5 with a Latin competitor and O = 2²⁰. At t=0 the cue can never supply 20 bits (e saturates at 12.4), so k* = 11, the Latin MAP value. At t=1 the cue must supply 7.6 bits, and e(6) = 7.26, e(7) = 8.49, which predicts 7; MC gives 6. At t=2 the odds are below 1, so k* = 4. All agree with §4.13 within ±1 **[HEURISTIC]**.

*Outcome.* An ideal Bayesian compounds substantially only if its prior odds against the environment's class exceed the cue's own evidence. It then saturates after about log₂O/λ tables, as a staircase whose step width is fixed by λ, a computable property of the environment. Graded compounding over t = 1..4 at n = 6 (λ ≈ 20 bits vs Latin) would need log₂O ≈ 40–80 under a Latin competitor. Under a loop competitor (λ ≈ 7) it would need ≈ 15–30. This line refines the program thesis rather than refuting it. Unbounded learners do compound, but only by log₂(1/π) bits, at a rate set by λ. Large observed compounding means a strongly mis-weighted or truncated t=0 hypothesis space, which is the resource-bounded case.

### 5.2 Line 2: notebook vs abstraction at equal resources

**Prop B (forced-cell monotonicity) [PROVED].** Let C′ ⊆ C be relabelling-closed classes, K a cue of T ∈ C, and suppose some table of C′ agrees with K. If cell c is C-forced, every C′-table consistent with K has value T(c) at c, because those tables are among the C-tables consistent with K. So the possibilistic C′-learner is correct on c with probability 1.

**Corollaries.**
- **(B1)** After any non-empty history of group tables, A's class is cyclic, abelian or group, with fallback. A is therefore correct on *every* group-forced (L ∪ O ∪ G) cell, whatever types the history contained. Over-specialisation, such as inducing "cyclic" from Z6-only histories, can only cost accuracy on U-cells. MC confirms 1.00 everywhere it applies.
- **(B2)** If n is prime, the only group of order n is Z_n. After t ≥ 1 group tables, N_iso's hypothesis set is Iso(Z_n), which is exactly A's class. The MAP weights are equal too (uniform over the same set). So **N_iso ≡ A on every cue, cell and t ≥ 1 at n = 5, 7.** MC agrees: identical rows in T1 and T3.
- **(B3)** For a held-out target type H*, N_iso is correct on G-cells when the cue is consistent with a seen type, which happens with probability q. Otherwise it falls back to Latin, where a G-cell has at least two feasible values. So acc_G(N_iso) ≤ q + (1−q)/2, while acc_G(A) = 1. MC at the design k gives q ≈ 0.01 at n=6.

**Missing-mass law [PROVED as an identity; the approximation h_H ≈ a₀ is HEURISTIC].** Let target types have frequencies p_H and history types be i.i.d. from p. Then, exactly,

  acc_G(N_iso, t) = Σ_H p_H [ 1 − (1−p_H)^t + (1−p_H)^t · h_H ],

where h_H is N_iso's G-accuracy on type H when the history misses H. With h_H ≈ a₀ (valid when q ≈ 0), the A − N_iso gap is (1−a₀)·Σ_H p_H(1−p_H)^t. That sum is the expected Good–Turing missing mass of the target's type. §4.8 matches it within 0.01 at n=6 and n=8. *Outcome:* the exemplar learner's compounding rate is set by a measurable property of the environment (its type-frequency distribution), while A's is one-shot. Baseline separation is impossible at prime n (B2) and weak at n = 4, 6 (m = 2: the gap at t=4 is 0.04). It is large at n = 8 (m = 5: gap .31 at t=4) and on held-out types (gap ≈ .7). This is the precise sense in which the brief's expected feature ("N stays near Latin, A drops") is necessary but not sufficient.

**Where this line fails.** The A→B→C curriculum barely separates learners on forced cells, by B1: at n=6, k=10 every learner except N and N_iso scores 1.00. Separation appears only on the commutative-copy probe at small k (§4.11). That probe separates *recency-based* induction (A_rec, Bh_rec: .84 at k=3) from *cumulative* induction (A_cum .47, Bh .52). It does *not* separate exemplar from abstraction (N_iso .47).

### 5.3 Line 3: teaching sets (the brief's k* as md)

**Prop A (lower bound) [PROVED].** For n ≥ 5 and any relabelling-closed class C ⊇ Iso(T), every defining cue mentions at least n−1 distinct labels. Hence md_C(T) ≥ ⌈(n−1)/3⌉.

*Proof.* Suppose two labels x ≠ y occur nowhere in cue K (not as row, column or value). Let τ be the label transposition (x y) and T′ = τ·T, meaning T′(τa, τb) = τ(T(a,b)). Then T′ ∈ C. For a cue cell (a,b) with value v = T(a,b), none of a, b, v is x or y, so T′(a,b) = τ(T(a,b)) = v, and T′ agrees with K. If K is defining, T′ = T, so τ is an automorphism of the labelled table. That gives a group automorphism α moving exactly two elements. Fix(α) is a subgroup of order n−2, so (n−2) | n, which forces n ≤ 4. Therefore, for n ≥ 5, at most one label is unmentioned. Each cell mentions at most 3 labels, so 3|K| ≥ n−1. ∎

The bound is attained by Q8 under the relabelling-only prior (3 cells), so it is tight at n=8. It is far from tight under the group prior.

**Outcome.** ⌈(n−1)/2⌉ is exact for all groups of order 2–7 and wrong at 8–9 (§4.2). The excess of the group prior over relabelling-only (0–2 cells: V4 +1, Z4×Z2 +1, Z2³ +2, D4 +1, Q8 +2, Z3×Z3 +1) is the price of excluding other isomorphism types. It is largest for types with many automorphisms or near-neighbours.

**Failure.** I found no closed formula. Z9 needs 5 cells even when its type is known, so md is not a function of n alone even for cyclic groups. An upper-bound construction of the form ⌈(n−1)/2⌉ + O(rank) was not attempted to a proof.

### 5.4 Predicted-outcome table, Q2 (learner × condition × measure)

Design point n=6, k=10 unless marked. G = associativity-forced cells, O = identity/cancellation-forced cells. "t0→1→4" lists values at t = 0, 1 and 4. Each entry is a sentence of the form "measure: value".

**R**
- Base G-accuracy (t0→1→4): .17 → .17 → .17.
- k*_0.9 (t=0 / t=4): none / none.
- Scramble G (t=4): .17. Held-out S3 G (t=4): .17.
- Loops, n=5 (G / O, t=4): .20 / .20.
- Lesion stage-2 G: .17.
- Copy-error, k=4, abc history: —. Z5 after loops, k=6: —.

**Np / N**
- Base G-accuracy: .30 → .30 → .30. k*_0.9: 20 / 20.
- Scramble G: .30. Held-out S3 G: .30.
- Loops, n=5: .33 / .35.
- Lesion stage-2 G: .30 (P1 = 0).
- Copy-error: .16.

**N_iso**
- Base G-accuracy: .30 → .65 → .96. k*_0.9: 20 / 8, with the gradual path 17, 15, 10.
- Scramble G: .30. Held-out S3 G: **.31**.
- Loops, n=5: .23 / .54.
- Lesion stage-2 G: .61.
- Copy-error: .37.

**A**
- Base G-accuracy: .30 → 1.00 → 1.00. k*_0.9: 20 / 7, a **step** at t=1.
- Scramble G: .30. Held-out S3 G: **1.00**.
- Loops, n=5: .46 / 1.00.
- Lesion stage-2 G: .95.
- Copy-error: .36 (A_rec .61).
- Z5 after loops: A_w 1.00.

**Loop learner (no associativity)**
- O = 1.00, G = .42 at any t.

**Asl**
- Base G-accuracy: .30 → 1.00 → 1.00. k*_0.9: 20 / 7.
- Scramble G: .30. Held-out S3 G: 1.00.
- Loops, n=5: **1.00 / 1.00**.
- Lesion stage-2 G: .95.

**B**
- Base G-accuracy: 1.00 flat. k*_0.9: 6 / 6.
- Scramble G: 1.00. Held-out S3 G: 1.00.
- Loops, n=5: 1.00 / 1.00.
- Lesion stage-2 G: 1.00.

**Bh**
- Base G-accuracy: 1.00 flat. k*_0.9: 7 / 6.
- Scramble G: **Latin level**; it falls to .40 at n=5, Nb, because only the Latin environment survives.
- Held-out S3 G: 1.00.
- Loops, n=5: **.47** / 1.00.
- Lesion stage-2 G: 1.00.
- Copy-error: .29 (Bh_rec .61).
- Z5 after loops: Bh_w **.82**.

**Which pattern shows that an abstraction changed learning itself, rather than being looked up?** The joint pattern below, measured against matched-length scramble controls:
1. A step in G-cell accuracy at t=1, not a coverage-gradual rise that follows the missing-mass curve.
2. Held-out-type transfer on G-cells equal to seen-type gain (abstraction fraction w = held/comp ≈ 1).
3. After group-like loops, G-cells drop to loop level while O-cells stay at ceiling. This is retraction of associativity specifically, absent in N/N_iso, and in Asl it shows as *no* drop.
4. The lesion keeps the gain when the model's own completion is correct, with high P1.

Exemplar lookup predicts gradual compounding, w ≈ 0, and no O/G dissociation. A model that has only the In-Context-Algebra mechanisms predicts O ≈ 1, G at loop level (.42), and no G-cell compounding at all.

### 5.5 Q3: pre-registration

**Stimuli.** `code/stimuli.py` emits JSONL trials: the prompt text, every queried cell, the gold word, and that cell's exact forcing class.

Format:
- facts `w1 * w2 = w3` in random order, tables headed `Table A:`, `Table B:`, …;
- each query appends `\nwa * wb =`;
- scoring is forced choice over the n target words by next-token log-probability of " word".

Example: `python stimuli.py --n 6 --cond heldout --t 2 --k 8 --trials 3 --out ../out/stim_test_heldout.jsonl` produced an S3 target with 14 G-cells and 14 O-cells.

**Token-leak controls (implemented):**
- C1: fresh vocabulary per table, with no word reused within a trial.
- C2: a random word↔element bijection per table; words drawn uniformly per trial.
- C3: random fact order and neutral headers.
- C4: blacklist of numerals, ordinals and identity-suggestive words.
- C5: a matched-length scramble control (identity-free Latin squares on fresh words, same fact count).
- C6: a recall positive control ("leak": the history repeats the target's labelled table).
- C7: `filter_vocab(tokenizer)` keeps only words that are a single token with and without a leading space. Run it with the Qwen and Llama tokenizers before generating.

Recommended additions: run each trial under two independent vocabulary draws and require agreement, and test for semantic-cluster effects by regressing accuracy on word-pair embedding similarity.

**Design cells** (n=6, k=10 unless noted; t is the number of complete history tables, about 216 tokens each):
1. base t ∈ {0, 1, 2, 4};
2. scramble t=4;
3. held-out (Z6-only history, S3 target) t ∈ {1, 4};
4. loops, n=5, k=6, t=4, with a Z5 target and with a loop target;
5. abc t=4 with commutative-copy probes at k=4;
6. lesion stage 2 (k1 = k2 = 10);
7. leak t=1;
8. optional n=8 base and held-out, k=16, t ∈ {0, 1, 4}: the strongest exemplar-vs-abstraction contrast (m=5);
9. for (iv), a small model fine-tuned on random group tables (fresh words), tested at t ∈ {0, 4} with loop histories.

**Numbers.** 200 trials per cell, about 13 cells, so about 2600 prompts per model, each carrying all unseen cells as queries (26 at n=6). This gives ≥ .99 correct classification under lapse ≤ .4 and ρ ≤ .6 (§4.14). 100 trials per cell is the minimum (≥ .97).

**Decision rule** (thresholds fixed in advance; `power.py`):
- Raw accuracy on Latin-forced cells at t=0, pooled: if acc_L < 1/n + .25, the model is **R**.
- Otherwise rescale every accuracy x to g(x) = (x − 1/n)/(acc_L − 1/n).
- If g(G at t=0) ≥ .85, the model is **B-type**, with the prior already in place. Then use the loops and (iv) cells to split rigid B/A_w (no drop) from revisable Bh (drop to loop level).
- Otherwise let base = max(g(G0), g(G_scramble,4)) and comp = g(G4) − base. If comp < .15, the model is **N**.
- Otherwise let held = g(G_heldout,4) − base. If held ≥ .6·comp, the model is **A**; else it is **N_iso**.
- Always report the continuous abstraction fraction w = held/comp and the shape statistic g(G4) − g(G1). Shape near 0 indicates A; for N_iso it is ≈ .3 at m=2.
- Secondary: loops O/G dissociation (A vs Asl vs N_iso); copy-error at k=4 (recency vs cumulative).

Every outcome is informative. Each branch names a learner class, and an intermediate w measures the mixture.

### 5.6 Results assumed from other candidates [CONDITIONAL]

- **Hypothesis H1** (from a candidate proving a notebook-vs-abstraction separation under bounded inference): matching a k-cell partial table against a stored Cayley table up to relabelling cannot be done by a depth-d transformer within budget β < n!/|Aut|. Under H1, N_iso is unavailable to fixed-size models at large n. The baseline condition at prime n then *does* separate notebook from abstraction (any compounding implies A), and the held-out cells become a confirmation rather than a necessity.
- **Hypothesis H2**: a theorem bounding the compounding of resource-bounded learners by the description length of the induced rule. With H2, the predicted step size in §5.1 becomes a lower bound on model capacity used. Neither hypothesis is assumed anywhere else in this report.

## 6. Referee verdicts

- **md correction.**
  - Objection: the claim was only "for 2–7", so nothing is wrong. Answer: correct, and the report says so. The new content is n=8–9 plus the bound.
  - Objection: is it computed correctly? Two independent exhaustive algorithms agree (DFS over all cues; ILP hitting set with certified optimality), and the minimum cues were rechecked directly.
  - Objection: is it known? Cayley-table reconstruction studies holes (Dénes; Vojtěchovský 2007), not minimum defining sets relative to the class of group tables; nothing found (§7).
  - **Verdict: survives.** Lower bound ⌈(n−1)/3⌉: survives (a complete proof, tight at Q8).
- **Prop C (B flat).**
  - Objection: true by definition. **Verdict: fails as a result.** It survives as a definitional repair, reported as such.
- **Prop D / clock.**
  - Objection: the mixture bound and Bayesian updating are textbook, and Wakayama & Suzuki 2026 and Tripuraneni et al. 2023 report fast or near-free task-family selection.
  - **Verdict: survives in a weaker form.** The principle is KNOWN. The new part is the paradigm-specific constants (λ table, e(k)) and the testable staircase prediction. The k* numbers are MC, not exhaustive.
- **Prop B and the B2 identity.**
  - Objection: near-definitional once N_iso is defined.
  - **Verdict: survives as a design-critical negative result.** It shows the brief's Q1 discriminator cannot detect "retrieval masquerading as abstraction" at n = 5, 7, which the brief itself named as the barrier.
- **Missing-mass law.**
  - Objection: an identity plus an approximation that fails at n=4, k=4.
  - **Verdict: survives in the weaker form stated.** It is exact with h_H, and the approximation is validated only at n=6 and n=8 at the design k.
- **Predicted-outcome table.**
  - Objection: idealised possibilistic and MAP learners are not LLMs, which will mix strategies and make content-independent errors.
  - **Verdict: survives as predictions for the idealised classes only.** The lapse-normalised rule and w are the bridge, but real models may sit off every idealised point. The power analysis assumes binomial-beta noise.
  - Objection: MC error. An exhaustive check at n=4 agrees within .02, and the decisive contrasts (1.00 vs .31) dwarf MC error.
- **"Beyond computed range?"**
  - md: no extrapolation is claimed (the formula is refuted).
  - Prop A, B and C and the missing-mass identity hold for all n.
  - Curve values are computed only for n ≤ 8.

## 7. Novelty check

**Searches run (WebSearch):**
- in-context learning of group operations / Cayley tables / arbitrary symbols;
- minimum defining set of a Cayley table among groups;
- partial Cayley table uniquely completable to a group;
- teaching dimension of Cayley / multiplication tables;
- "how many entries of a group multiplication table determine the group table";
- "defining set / critical set group table uniquely determines";
- smallest critical set of back-circulant squares;
- transformer in-context meta-learning across tasks;
- Latin-square completion by LLMs.

**Papers opened, and how close each comes:**
- Todd, Brinkmann, Gandikota, Bau, *In-Context Algebra*, https://arxiv.org/abs/2512.16902 (HTML v2 read). Closest on the task: per-sequence symbol assignment, mechanism-targeted distributions, held-out order-8 groups. No cross-table compounding, no forcing-class fingerprint, no pretrained LLMs.
- Hatami & Qian, *Teaching dimension, VC dimension, and critical sets in Latin squares*, https://arxiv.org/abs/1606.00032. Identifies Latin critical sets with teaching dimension. Our md is the analogue for the class of labelled group tables, which they do not treat.
- Cavenagh 2006, lower bound for critical sets of the back-circulant square, https://researchcommons.waikato.ac.nz/entities/publication/7321fe42-c570-4f52-b416-915e88a187ac (abstract page).
- Vojtěchovský 2007, *Reconstruction of group multiplication tables by quadrangle criterion*, https://arxiv.org/abs/math/0701697. On holes; no minimum defining sets (checked in the PDF text).
- Wanless & Webb, partial Latin squares not embeddable in a Cayley table, https://arxiv.org/abs/1504.08101.
- Raventós, Paul, Chen, Ganguli 2023, https://arxiv.org/abs/2306.15063, and Wurgaft et al. 2025, https://arxiv.org/abs/2506.17859. Memorizing (discrete prior on seen tasks) vs generalizing predictors: the formal ancestor of N_iso vs B/A. The missing-mass law is the fresh-label, in-context analogue; I did not find it stated.
- Chan et al. 2022, https://arxiv.org/abs/2210.05675. In-context information gives exemplar-based generalisation, in-weights information rule-based. This predicts N_iso-like behaviour in context and A_w in weights, which is condition (iv).
- Tripuraneni, Doshi, Yadlowsky (NeurIPS 2023 workshop page), https://neurips.cc/virtual/2023/80529, and Wakayama & Suzuki (ICML 2026 poster), https://icml.cc/virtual/2026/poster/65655. These are near-misses for Prop D: task-family selection is nearly free, and mixture uncertainty vanishes quickly.
- Hendel, Geva, Globerson 2023, https://arxiv.org/abs/2310.15916; Todd et al. 2023, https://arxiv.org/abs/2310.15213; Olsson et al. 2022, https://arxiv.org/abs/2209.11895; Xie et al. 2021, https://arxiv.org/abs/2111.02080; He et al. 2024, https://arxiv.org/abs/2406.02550; Yang et al. 2025, https://arxiv.org/abs/2502.20332.
- OEIS A002860 (https://oeis.org/A002860) and A034383 (https://oeis.org/A034383) for the counts.
- Dénes' hole-reconstruction theorem is known to me only through search snippets: **unverified**.

**Verdict.** The md values beyond order 7, and the lower bound, were found nowhere (novelty moderate). The design-level results are applications of known principles to a new paradigm.

## 8. Reach (predictions outside the home domain, concrete enough to fail)

1. **Other fresh-label ICL tasks** (In-Context-Algebra-style models; permuted-symbol modular arithmetic).
   - Models trained with only cyclic groups should show copy errors (answering T(b,a)) on dihedral targets at small cue sizes, at rates near our abelian-history values (≈ .6–.85 at 3–4 revealed facts). The rate should fall to ≈ 0 once the cue contains one non-commuting pair.
   - Their compounding across several groups in one sequence should be a step (A) if they rely on cancellation and identity rules. It should follow the missing-mass curve if they rely on group-specific exemplars.
2. **Human one-shot rule learning** (Latin-square or "secret operation" puzzles).
   - Participants shown one complete Z6 table should complete held-out S3 cue cells forced by identity and cancellation (O-cells) at ceiling.
   - Associativity-forced G-cells should stay near the loop-learner level (≈ .4) unless associativity is explicitly taught, because associativity-based inference needs multi-step search.
   - Exemplar-strategy participants should show coverage-gradual improvement across tables of different types.
3. **Program synthesis / library learning.**
   - With libraries induced from problems of one "type", gains on new problems sharing the axioms but not the surface type separate library learning (A) from retrieval-augmented synthesis (N_iso).
   - The retrieval system's success rate on new problems should track the Good–Turing missing mass of problem types in its memory.
3. **Pretraining / fine-tuning.**
   - Fine-tuning on random group tables should make G-cell accuracy at t=0 approach the B level and remove in-context compounding.
   - Whether in-context loop tables then *lower* G-cell accuracy (Bh_w: about −.18 at n=5, k=6) or not (A_w) tests whether the in-weights abstraction is revisable.

## 9. Research log

- **Proved:**
  - Prop A: md lower bound ⌈(n−1)/3⌉, n≥5.
  - Prop B: forced-cell monotonicity; B1, B2 (N_iso ≡ A at prime n), B3.
  - Prop C: B is t-flat.
  - Prop D: mixture bound, applied.
  - Prop E: the clock identity.
  - The missing-mass identity.
- **Verified (exhaustive):**
  - md values for every group of order ≤ 9, under the group, abelian, cyclic and relabelling priors (plus loop ≤ 6, Latin ≤ 5);
  - the n=4, k=4 learner accuracies;
  - counts.
- **Heuristic (MC, exact per cue):** all learner curves, k*, the condition tables, the clock numbers, and the power analysis.
- **Refuted:** ⌈(n−1)/2⌉ for all groups (from n = 8).
- **Failed or weak lines:**
  - A→B→C is weakly diagnostic on forced cells (B1). Fix: the commutative-copy probe at small k.
  - The absolute-threshold decision rule fails under lapses. Fix: normalise by Latin-forced accuracy.
  - B_hier with a balanced hyperprior shows no compounding, so a Bayesian compounding curve needs prior odds ≥ 2^λ.
  - n ≤ 4 cannot test associativity.
  - The Latin MAP learner and B_hier are not computable at n ≥ 6–7 (no enumeration). Possibilistic Latin was used instead.
  - Exhaustive enumeration over cues is infeasible at the design sizes (C(36,10) ≈ 2.5·10⁸).
  - No closed formula for md.
  - Order 10 (Z10, D5, 1088640 labelled tables): the ILP was still running at writing time. The first relaxation gave a 5-cell candidate with 4 violating tables, so md(Z10) ≥ 5. If finished, the result is in `out/md_n10.json` and `out/log_md10.txt`.
- **Bugs found and fixed:**
  - MAP fallbacks at n ≥ 6;
  - automorphism closure in `md_extra.py`, validated afterwards by exact set equality against brute force at n = 6, 8;
  - the static B crash on loop targets.
- **Open next steps:**
  1. Run `stimuli.py` with the Qwen/Llama tokenizers and the decision rule.
  2. Prove or refute md(Z_n) ≥ ⌈(n−1)/2⌉ for cyclic groups (true for n ≤ 9).
  3. Extend the held-out design to n=8 with exact per-type cue calibration.
  4. Fit the compounding clock to model data by estimating prior odds O from the t=0 snap point and predicting the t=1, 2 steps.
