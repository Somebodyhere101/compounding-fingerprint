# Candidate 10 log: compounding fingerprint for transformers

Start: 2026-10-06 16:27:31 (local). Soft budget ~3h.

Python: C:\Users\%USERNAME%\AppData\Local\Temp\ccv\Scripts\python.exe
Base dir (BASE): ...\runs\c10 ; code in code\, outputs in out\

## Plan
1. Literature check (background items; ICL algebra/groups; task/function vectors; induction heads; Bayesian ICL; critical sets / defining sets; counts of Latin squares and labelled groups).
2. Verify brief claim: min group-prior defining set = ceil((n-1)/2) for every group of order 2..7 (exhaustive).
3. Implement exact learners R, N (surface + bounded Latin propagation; N_iso exemplar matcher), A (axiom induction), B (true prior) and B_hier (hyperprior over classes). Compute k*(t), per-class accuracies, n<=5 exact, n=6,7 where feasible; conditions baseline, lesion, wrong abstraction (loops), A->B->C, in-weights prior.
4. Predicted-outcome table + decision rule + power analysis.
5. Stimulus generator with token-leak controls.
6. Report.

## Hypothesis tree (initial, confidence)
- H1 (90%): B with the TRUE prior has a t-independent fingerprint under fresh labels (conditional independence) -> no compounding by definition; the right Bayes benchmark is hierarchical (B_hier).
- H2 (70%): B_hier's compounding is small (bounded by log2(1/p) bits per target, mixture bound), so a large k* drop with t marks a resource-bounded learner.
- H3 (60%): N_iso (exemplar matching) and A are indistinguishable on the baseline condition once all iso classes of order n have been seen (n<=7 has <=2 classes); separation needs held-out iso classes (Z4/V4, Z6/S3, n=8 classes) or misleading curricula.
- H4 (50%): brief's ceil((n-1)/2) claim may fail for V4 (hand analysis suggests 2 cells may not suffice).

## Notes / computations

### 16:30-16:50 Counts and minimum defining sets (md)
- `python code/tables.py`: labelled group tables n=1..8: 1,2,3,16,30,480,840,22080 (= OEIS A034383); Latin n<=5: 1,2,12,576,161280; labelled loops n<=6: 1,2,3,16,280,56448 (assoc ones = groups).
- `python code/defsets.py 2,3,4,5,6,7 ../out/defsets.json` (bitset DFS, exhaustive):
  group prior md: Z2 1, Z3 1, Z4 2, V4 2, Z5 2, Z6 3, S3 3, Z7 3  -> brief's ceil((n-1)/2) CONFIRMED for orders 2..7.
  Latin prior md (n<=5): Z3 2, Z4 4, V4 5, Z5 6. Loop prior: Z5 3, Z6 5, S3 6.
- `python code/defsets.py 8 ../out/defsets8.json`: group prior md: Z8 4, Z4xZ2 5, Z2^3 6, D4 5, Q8 5 ; abelian prior: Z4xZ2 4, Z2^3 4.
  => ceil((n-1)/2)=4 FAILS at n=8 for 4 of the 5 groups.
- `python code/defsets_ilp.py 2,3,4,5,6,7,8 group ../out/md_ilp_2to8.json` (independent: min hitting set, CP-SAT, lazy constraints): identical values.
- `python code/defsets_ilp.py 9 group ../out/md_ilp_9.json`: Z9 5, Z3xZ3 5 (ceil(8/2)=4 fails); `python code/check9.py`: DFS finds no defining cue of size<=4 for either; ILP 5-cues each leave exactly 1 table.

### Literature (running)
- In-Context Algebra, Todd, Brinkmann, Gandikota, Bau, arXiv 2512.16902 (ICLR 2026): https://arxiv.org/abs/2512.16902 ; small 4-layer transformers trained on sequences "vx vy = vz" with per-sequence random variable<->element map, cyclic+dihedral groups up to order 10, several groups per sequence with disjoint variables; generalise to held-out order-8 groups; mechanisms: commutative copying, identity recognition, closure-based cancellation; associative composition only 60.2%. No pretrained LLMs.
- Raventos, Paul, Chen, Ganguli 2023, arXiv 2306.15063 https://arxiv.org/abs/2306.15063 : below a task-diversity threshold transformers act as Bayes with discrete prior on pretraining tasks (notebook-like), above it like ridge (general prior).
- Wurgaft et al. 2025, arXiv 2506.17859 https://arxiv.org/abs/2506.17859 : memorizing predictor (discrete prior on seen tasks) vs generalizing predictor (true prior); hierarchical Bayesian account with loss-complexity tradeoff. = closest formal analogue of N_iso vs B.
- Hendel, Geva, Globerson 2023 task vectors https://arxiv.org/abs/2310.15916 ; Todd et al. 2023 function vectors https://arxiv.org/abs/2310.15213 ; Olsson et al. 2022 induction heads https://arxiv.org/abs/2209.11895 ; Xie et al. 2021 implicit Bayes https://arxiv.org/abs/2111.02080 ; He et al. 2024 modular arithmetic ICL task-number transition https://arxiv.org/abs/2406.02550 ; Chan et al. 2022 in-context=exemplar vs in-weights=rule https://arxiv.org/abs/2210.05675 ; Yang et al. 2025 symbol abstraction/induction/retrieval heads https://arxiv.org/abs/2502.20332 ; Wakayama & Suzuki ICML 2026 (task-mixture uncertainty vanishes exponentially fast with few in-context examples) https://icml.cc/virtual/2026/poster/65655
- Critical sets: Hatami & Qian 2016, "Teaching dimension, VC dimension, and critical sets in Latin squares" https://arxiv.org/abs/1606.00032 (scs = min teaching dimension of Latin squares; Nelder conjecture floor(n^2/4); lower bound n^2/10^4). Cavenagh 2006 AJC lower bound for back-circulant critical sets (n^{4/3}/2 - ...) https://researchcommons.waikato.ac.nz/entities/publication/7321fe42-c570-4f52-b416-915e88a187ac . Vojtechovsky 2007 reconstruction of Cayley tables with <= n-1 holes by quadrangle criterion https://arxiv.org/abs/math/0701697 (holes, not entries).
- No source found that defines min defining sets of a group table relative to the class of GROUP tables (searched 3 phrasings).

### 16:52-17:05 Learner simulations (sim.py) and Bayesian evidence (evidence.py)
- code/engine.py (per-cue exact counts; SAT feasibility for Latin/loops n>=6, validated vs enumeration: `python code/test_engine.py` 0 mismatches n=4,5,6).
- code/sim.py n M kstep: learners R, Np (Latin propagation only), N (Latin feasibility), Nb (Latin MAP), Niso/Nisob (exemplar iso-matching), A/Ab (axiom induction, strict, fallback chain), Asl (sloppy: identity => group), Bposs/B (true prior), Bh (hierarchical Bayes, uniform hyperprior over latin/loop/group/abelian/cyclic envs available at that n).
- Bug found+fixed 16:58: for n>=6 Nisob/Ab were undefined on cues where they fall back to a SAT-only class -> biased averages. Fixed (fallback = possibilistic); n=6,7 rerun (M=150). n=4,5 unaffected (Latin enumerable).
- n=6 (first run, M=100) k*_0.9(t=0..4): N 21/19 flat; Niso S3 21,19,14,10,8 (gradual = coverage 1-2^-t); A 21,7,7,7,7 (step); B 6/7 flat; Bh 8,7,6,6,6 (S3) and 6,7,7,7,7 (Z6). Heldout S3 (Z6-only history): Niso flat 21, A 8.
- evidence.py a: per-table evidence D(groups||Latin) bits: Z4 4.58, V4 6.17, Z5 12.39, Z6 20.11, S3 21.69, Z7 36.09; D(groups||loops): Z4 -0.58, V4 1.00, Z5 3.22, Z6 6.29, S3 7.88, Z7 17.11.
- evidence.py c 6 40 (broad env = loops): k*(t=0,1,2) for log2 prior odds 0/5/10/20/40: Z6 [7,7,7] [9,7,7] [12,8,7] [12,12,10] [13,12,12]; S3 [7,7,6] [9,7,6] [9,7,6] [9,9,9] [12,9,9]  -> Bayesian compounding needs ~log2(O)/D tables ("compounding clock").
- n=5 cue evidence e(k)=E log2 P_grp(K)/P_latin(K): k=2 0.77, 4 3.91, 6 7.26, 7 8.49, 10 11.63, ->12.39; vs loops ->3.22.

### 17:05-17:20 stimuli, report tables, power
- code/stimuli.py: generator (conditions base|heldout|loops[--loop_target]|scramble|abc|leak|lesion1; lesion_stage2()); exact forcing class per query; token-leak controls C1-C7. Test: `python code/stimuli.py --n 6 --cond heldout --t 2 --k 8 --trials 3 --out ../out/stim_test_heldout.jsonl` -> S3 target, k=8: 14 G + 14 O cells, 0 L, 0 U.
- code/report_tables.py: env-averaged tables T1-T6.
- code/power.py: decision-rule confusion. Absolute thresholds FAIL for B under lapses (lapse .2 -> B misread as N). Fixed by lapse-correction g(x)=(x-1/n)/(accL-1/n). With correction: correct classification >=0.97 at N=100 trials/cell for lapse<=0.4, rho<=0.6; >=0.988 at N=200.
- `python code/md_extra.py iso` (labelling-only prior, ILP): Z2 1, Z3 1, Z4 2, V4 1, Z5 2, Z6 3, S3 3, Z7 3, Z8 4, Z4xZ2 4, Z2^3 4, D4 4, Q8 3, Z9 5, Z3xZ3 4.  Q8 attains the proved lower bound ceil((n-1)/3)=3. Group-prior excess over labelling-only: V4 +1, Z4xZ2 +1, Z2^3 +2, D4 +1, Q8 +2, Z3xZ3 +1, others 0.
- 17:20 `python code/abc_inweights.py w 40` -> out/inweights_n5.json ; `python code/summ_abc_w.py`:
  Z5 target: A_w (rigid in-weights group prior) k*=4 for all t, any history; Bh_w (revisable, hyperprior 2^10:1 for groups) k*=4 with group history, 4->7 after >=1 loop table (acc@k=6: 1.00 -> 0.82): in-context loops override a revisable in-weights prior.
  loop targets: A_w k*=8; Bh_w k*=8 (acc@k=6 0.83 -> 0.87 after loops).
- abc run crashed (B static fallback None on loop targets); patched sim.py (B falls back to loop/Latin feasibility when the true-prior class has no consistent table); rerun queued with n=8 sim (M=20, kstep 4).
- 17:22 `python code/exact_small.py 4 4` (ALL 1820 cue sets per target): base O-cells t=0,1,2: N .4289 .4289 .4289; Niso .4289 .8015 .9008; A .4289 1 1; B 1 1 1; Bh 1 1 1. all-cells B .9544. heldout(V4, Z4-only history) O-cells: Niso .4301 .7076 .7076; A .4301 1 1. Matches MC (M=400) within ~0.02.
- Literature add: Tripuraneni, Doshi, Yadlowsky NeurIPS 2023 "Can Transformers In-Context Learn Task Mixtures?" https://neurips.cc/virtual/2023/80529 (task-family selection cost near zero when pretraining balanced; complex non-monotone curves for rare families) ~ consistent with our B_hier result. Wanless & Webb 2015/2017 small partial Latin squares not embeddable in any Cayley table https://arxiv.org/abs/1504.08101 .
- 17:25 `python code/abc_inweights.py abc 40` (k step 2): S3 target after [2 mixed + 2 Z6]: k*_0.9 N 24, Niso 16, A_cum 8, A_rec 8, Bh 6, Bh_cp 8; at k=10 A/Bh all 1.00 on O/G cells (Prop 2: over-specialisation harmless on forced cells). Group-like loop targets (assoc frac .875): N 18, Niso 18, A 14, Bh 14.
- `python code/ccopy.py 150` commutative-copy probe (S3 target; probe cells: (b,a) revealed, T(a,b)!=T(b,a)); copy-error rate at k=3/4: N .16/.16; after abelian-only history A,Bh,Niso .84-.85/.61-.66; mixed t=4: A .37/.30, Bh .44/.22, A_rec .68/.51; abc: A_cum .47/.36, A_rec .84/.61, Bh .52/.29, Bh_rec .84/.61, Niso .47/.37. => copy-error rate at small k separates recency-based from cumulative abstraction; not exemplar vs abstraction.
- 17:30 `python code/evidence.py cL 5 60` (competitor = Latin only, lambda = 12.39 bits/table): k*(t=0,1,2) for log2 odds 0/5/10/20/40: [4,4,4] [4,4,4] [7,4,4] [11,6,4] [11,11,11]. Clock check: O=2^20,t=1 needs 20-12.4=7.6 bits from the cue; e(6)=7.26,e(7)=8.49 -> predicted k*~7, observed 6 (approximate law, mean-evidence version).
- ~17:28 n=8 sim (M=20, k step 4) done: k*_0.9 Latin 36-44, group 8-16; A 12 at t>=1; Niso 40,40,40,36,36 (env avg); held-out D4/Q8 (abelian-only history): Niso flat (44/36), A 12; G-acc@k=16 Niso .22,.38,.51,.61,.69 vs A 1.00.
- `python code/coverage.py`: missing-mass law acc_G(Niso,t)=c_t+(1-c_t)a0, c_t=1-(1-1/m)^t: n=6 (m=2) sim .298/.653/.827/.913/.957 vs law .298/.649/.825/.912/.956; n=8 (m=5) sim .217/.380/.508/.609/.688 vs law .217/.373/.499/.599/.679. Fails at n=4,k=4 (q~0.49 not ~0).
- `python code/loop_learner.py 6 10 100` etc: loop learner (identity+cancellation, no assoc) O=1.00, G=.418 (n=6,k=10), G=.464 (n=5,k=6), G=.340 (n=7,k=12).
- md_extra.py automorphism bug (closure never expanded) found and fixed; coset-rep enumeration validated vs brute force at n=6,8 (exact set equality). n=10 run in progress.
- 17:31 n=5 sim done (M=80): baseline k*_0.9: N 11 flat; Niso = A = 11,4,4,4,4 (identical, as proved for prime n); B 4; Bh 4 flat. Wrong abstraction (t loops, Z5 target): A k* 11->8, G-acc@6 .33->.46 (loop level), O-acc ->1.00; Asl G 1.00; Bh G 1.00->.49 (negative transfer); Niso G .33->.23, O .35->.54; N flat .33. Lesion n=5 k1=6: P1 N .00 (stage-2 G .33), A .99 (stage-2 G .99).
- out/report_tables.md generated by `python code/report_tables.py 0.9`.

### 17:35-17:45 power rule update, report, n=10
- power.py: baseline for comp/held changed to max(G0, G_scramble4); Bh's scramble value = Latin level (only Latin env survives identity-free histories). Results (`python power.py > ../out/power.txt`): correct classification >= .88 (N=25), >= .94 (50), >= .97 (100), >= .99 (200) for lapse <= .4, rho <= .6.
- REPORT: the Write tool refused to create report.md ("Subagents should return findings as text, not write report files"). Complied: report content returned in the StructuredOutput instead; no workaround attempted. Tables for the report are in out/report_tables.md (generated by code/report_tables.py), power in out/power.txt.
- n=10: md_extra.py n10 (optimisation ILP) round 1 relaxation optimum k=5 (=> md(Z10) >= 5, valid lower bound since relaxation), round 2 still k=5 with violators; stopped (slow optimality proofs). Restarted as md10_fast.py (fixed-k feasibility, 20000 nearest tables initial, lazy violators).
- Conjecture (data n<=9): md_Iso(Z_n) >= ceil((n-1)/2) (Z9 has 5 > 4). Not proved; simple linear-algebra counting fails because non-bijective vectors can dominate the solution space.
- 17:47-17:51 `python code/md10_fast.py 4` (log out/log_md10_fast.txt): group prior, order 10 (1088640 labelled tables): Z10 k=4 infeasible (relaxation -> valid lower bound), k=5 feasible after 48 lazy rounds, cue {1*5=6, 1*9=0, 2*7=9, 4*3=7, 8*6=4}; D5 k=4 infeasible, k=5 feasible, cue {1*4=7, 2*4=8, 2*6=7, 3*9=1, 4*1=5}. `python code/check10.py`: each cue leaves exactly 1 of the 1088640 tables. => md(Z10)=md(D5)=5=ceil(9/2): formula HOLDS at n=10; exceptions within n<=10 are exactly orders 8 and 9.
- 18:00 `python code/md10_iso.py` (relabelling-only, order 10): Z10 k=3,4 infeasible, k=5 ok -> md_iso(Z10)=5 (consistent with conjecture md_iso(Z_n)>=ceil((n-1)/2)); D5 k=3 infeasible, k=4 running.
- 18:17 D5 relabelling-only: k=3 infeasible; k=4 undecided after >100 lazy rounds (stopped at final response) -> md_iso(D5) in {4,5} (<=5 from group-prior cue).
- FINAL STATE: all computations logged above; tables in out/report_tables.md, out/power.txt; report.md could not be written (harness refused report files for subagents) -> content returned via StructuredOutput.
- Elapsed ~1h50m.
