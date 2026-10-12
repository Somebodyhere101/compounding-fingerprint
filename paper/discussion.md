# Discussion and limitations

## What the study establishes
1. **A measurement that can tell abstraction from exemplar reuse in context.** Each queried cell is labelled exactly by the weakest axiom that forces it. Each history is controlled: a length-matched scramble, held-out isomorphism types, a recall leak control, and non-associative loops.
   - Together these give distinct, preregistered signatures for recall, Latin bookkeeping, exemplar matching, abstraction and Bayes-with-prior.
   - The decision rule recovers all seven preregistered exact learners, plus a post-hoc count-shortcut learner, with no error on the same trials the models see.
   - Only held-out transfer separates abstraction from exemplar matching, and at prime n the two are provably identical. Any claim of "learning to learn" an algebraic structure in context therefore needs composite orders and held-out types.
2. **A calibrated null for pretrained LMs up to 8B.** Nine models (ten model–precision configurations) from three families were tested: Qwen3 0.6–8B (base and post-trained), SmolLM2-1.7B and OLMo-2 (1B, 7B).
   - None infers in direct-answer mode. That includes the row/column elimination that would score 0.92 at the densest cue (a post-hoc baseline).
   - Of the eight calibrated models (all but post-trained Qwen3-0.6B), every one is non-inferential on this format under the calibration rule with the ≥30-query guard. The preregistered k=16 follow-up for the two models that qualify without the guard: both remain R at k=16, with acc_L 0.18 and 0.22. None compounds.
   - Recall of a previously seen labelled table approaches ceiling for Qwen3 at 4–8B (0.95–0.96), but not for OLMo-2 (0.31–0.37) or SmolLM2 (0.44).
   - In Qwen3, then, the absence of compounding is not a failure to read the context: the prerequisite inference is missing.
3. **A shortcut that trained networks find instead.** Small transformers trained from scratch on group, Latin and loop diets converge to a global count statistic: answer with the symbol least represented so far.
   - This matches the fingerprint accuracy of five of the six learned networks to within 0.001–0.007; the 6-layer network exceeds it by 0.014.
   - It accounts for the in-training plateau at n=6: the shortcut scores 0.33, and the networks 0.29–0.36 at step 14,000.
   - The fingerprint classifies them correctly, with the exact count learner (R).
   - The preregistered learning-failure threshold (chance + 0.10) is passed by this shortcut alone (0.33 at n=6). Future preregistrations of in-context algebra training should set "learned" against the count learner rather than against chance.
4. **A positive control on a real network.** The released In-Context Algebra model is classified non-R, non-B. Latin- and loop-forced cells score 0.89–0.99, and associativity-only cells 0.53–0.70, well above learner C and row/column elimination. Of the six released checkpoints scored (exploratory), those at 0 and 5k steps are R and those from 10k on are non-R.
   - The categorical classes miss its graded group prior. g(G t0) rises from 0.49 to 0.65 after the transition. Reporting g(G t0) as a continuous index is therefore more informative than the B-type cut-off alone.
5. **A validity limit for activation-space "mode" probes.** A published compute–retrieve axis, which separates its calibration prompts with AUC 0.96–0.99, does not track retrieval vs inference once format and length are matched. In three of five models it moves significantly in the opposite direction, and its sign flips with prompt format. An exploratory vocabulary-overlap control attributes 64–87% of the reversed gap to context vocabulary. With vocabulary matched, a retrieval signal of the predicted sign appears in the post-trained model (small, d_z +0.36, comparable to its vocabulary effect) and without the chat template (d_z +0.33 to +1.67). In base models under the chat template, a much larger surface component dominates: the vocabulary effect is 4–12× the retrievability contrast.

## What it does not establish
- **No claim about larger models or test-time reasoning.** The reasoning probe (Qwen3-1.7B, thinking on) was dominated by truncation: 28 of 30 thinking runs hit a 4,096-token budget, and the 2 that finished were correct. It neither shows nor excludes reasoning-enabled Latin inference. It also asks only whether a model can apply a *stated* rule, not whether it can induce one from histories.
- **The positive control is only partial.** The method-validation criterion of Experiment 3 is not met: A4 and I4 are R, not A/B-type and N_iso. The preregistered depth claim is not supported. The released In-Context Algebra model (Experiment 6) supplies a positive control for steps 1–3 of the rule: it is non-R, non-B. Its 16-token vocabulary rules out the compounding branch, so N vs N_iso vs A on a real network remains untested.
  - The cause lies in the training, not in the fingerprint. The networks learned a count shortcut, and the fingerprint reports exactly that.
  - Experiment 6 does not isolate which part of the training matters.
    - Parameters: the In-Context Algebra model has about 16× ours.
    - Tokens: its released checkpoints leave class R between 5k and 10k steps, after about 0.64–1.28B training tokens, comparable to our 0.92B. Token budget therefore does not explain the difference.
    - Recipe: it also differs in width (1024 vs 256), learning rate, vocabulary (16 vs 70 tokens), loss target (every token vs answers) and context density (groups-only, 200-fact sequences vs sparse cues).
  - Demonstrating the A signature on a trained network therefore needs a model of this kind with a vocabulary large enough for t ≥ 2 histories, or a curriculum that defeats the count shortcut at small scale. One example is a loss on cells where the least-revealed symbol is wrong.
  - The 6-layer network is the only one that exceeds the shortcut, by 0.014.
- **Quantisation.** The 4B, 7B and 8B models (Qwen3-4B-Base, OLMo-2-7B, Qwen3-8B-Base) ran in nf4 because of the 8 GB GPU. The preregistered bf16-versus-nf4 control (Qwen3-1.7B-Base) failed its tolerance: nf4 lowers G accuracy by 0.02–0.07 and changes 44% of per-query answers.
  - nf4 created no spurious effect, and it preserved strong context use.
  - Because nothing was there to hide at 1.7B, the control cannot rule out that nf4 hides a weak effect at 4B, 7B or 8B.
  - nf4 accuracies should not be compared directly with bf16 levels.
- **Format.**
  - Facts are unordered lines of single-token common English words used as arbitrary labels.
  - Row-sorting the target's facts helps at k=28: about equally in Qwen3-4B-Base and Qwen3-8B-Base (+0.10 to +0.11), and less (+0.005 to +0.046) in every other model, including the larger OLMo-2-7B (+0.045).
  - An instruction header does not help (−0.08 to +0.04, mean −0.006).
  - Other formats (grids, natural-language framing) may lower the onset scale. The fingerprint itself is format-independent, because classes depend only on the cue.

## Threats to validity, and how each was handled
- **Context length.** Every compounding contrast is against a scramble history matched in token count. Two reliable raw t=0 → t=4 increases disappeared against scramble: OLMo-2-1B at n=6 and Qwen3-0.6B at n=8.
- **Leakage through tokens.** Each table uses fresh single-token words, a random bijection and a semantic blacklist. All 2,800 Experiment 1–2 stimulus trials were re-verified independently by the audits: forcing classes were recomputed from scratch with independent SAT and enumeration code, with 0 mismatches. The 600 addendum-2d trials were generated afterwards. The same checker found 0 class mismatches over their 12,000 queries (`audit/sa_stim_k16.jsonl`).
- **Scoring artefacts.** Batched, offloaded and version-mixed scoring were each checked against sequential scoring:
  - argmax agreement was 92–98%, with near-ties in bf16;
  - the Qwen3-1.7B-Base primary contrast was re-scored with one code version (−0.006 vs −0.007);
  - classes are identical on accuracy and on p_gold.
- **Analysis flexibility.** The rule and thresholds were frozen before any output.
  - Experiment 3's allowed adaptations (a 2×-steps retrain of A4, an M=8 rerun of P4) did not trigger: A4 passed the learning rule, and P4 did not generalise.
  - Experiment 2's k=16 operating point was excluded by a ≥30-query guard that was added after the first sweep had been seen. Under the rule as frozen, it triggers for Qwen3-0.6B-Base and Qwen3-1.7B-Base (2 of 4 L queries correct at k=6), so it was run after the final audit: both remain R at k=16, with acc_L 0.18 and 0.22.
  - Unplanned changes are listed in Methods, "Deviations". They include:
  - training continuations after interruptions (A4, G4, A6) and a from-scratch retrain of A2;
  - the ≥30-query guard on the k_L rule, and the late addendum-2d run that followed from it;
  - the post-hoc sorted-format diagnostic;
  - scoring-code revisions, and CPU offload for the 7–8B models;
  - the Experiment-4 budget change;
  - the Experiment-5 query-count and interpretation deviations;
  - the post-hoc Experiment-6 mechanism split and loop-level baseline.

  Apart from the ≥30-query guard, none altered a stimulus, threshold or decision rule.
  - Post-hoc analyses (learner C, the shortcut diagnostic, the answer-bias diagnostic, Experiment 5b, the Experiment-6 mechanism split) are labelled, and none is used in a preregistered classification.
  - The shortcut diagnostic and learner C were introduced post hoc to attribute Experiment 3's method-validation mismatch to the training.
  - Learner C was fixed before Experiment 6 was preregistered, and it is that experiment's preregistered comparator (P4) and a secondary baseline.
- **Reporting errors.** Two adversarial audits (2026-10-07, 2026-10-09) re-derived every number reported for Experiments 1–5. Two later verification passes checked the revised text, and a third checked Experiment 6 (2026-10-11). Together they found and corrected these errors:
  - an inflated copy-bias statistic;
  - an unguarded partial-file value;
  - order-dependent bootstrap seeds;
  - stale ranges and model counts;
  - an unapplied learning-failure rule;
  - an overstated "dense-context Latin inference" reading of the trained networks;
  - a post-hoc guard that removed the preregistered k=16 follow-up for two models. That follow-up was then run: both remain R at k=16, with acc_L 0.18 and 0.22.

  None of these changed a fingerprint classification.

## Implications
For in-context learning research, "the model improves after seeing related examples" is weak evidence of abstraction unless all three hold:
1. the improvement survives a length-matched control;
2. the cells measured are forced only by the structure in question;
3. it transfers to held-out types.

"The model solves more cells as the cue grows" is weak evidence of constraint inference unless it beats a global count baseline.

The tools released here make these tests routine: the stimulus generator, exact forcing classes, idealised learners (now including C) and the decision rule. Experiment 6 shows the tools are not blind: on a network actually trained for the task they return a non-R class and an associativity deficit, consistent with that network's published mechanisms.

For the broader "experience compounding" question, these results place the phenomenon, if it exists for algebraic abstractions, beyond the scale and inference mode tested here. They also give a concrete, falsifiable target: the onset of sparse-cue Latin inference above the count baseline should precede any compounding, and compounding with w ≈ 1 should follow.

For representation-level claims about "modes" of processing, the CR result is a cautionary case. A direction that separates prompt *types* need not track the *process*. A format- and length-matched behavioural contrast, like leak vs base here, is a cheap test that any such probe should pass before it is interpreted.
