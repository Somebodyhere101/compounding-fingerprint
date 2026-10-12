# Abstract (draft)

Does a language model get better at a structured task because earlier examples taught it the structure, or because it reuses the examples themselves? We introduce a *compounding fingerprint* that separates these explanations exactly.

**Method.**
- Models complete partial operation tables (Cayley tables) after 0–4 earlier complete tables in the same context. The symbols are common single-token English words, freshly assigned as arbitrary labels in each table.
- Every queried cell is labelled by the weakest axiom that forces its value: Latin, loop, or group, the last forced only through associativity.
- Every history is controlled by a length-matched scramble, held-out isomorphism types, a recall (leak) control, and non-associative loops.
- A preregistered decision rule maps the resulting signature to one of several learner classes: recall, Latin bookkeeping, exemplar matching, abstraction, or Bayes-with-prior. On identical trials it recovers seven exact algorithmic learners, plus a post-hoc count-shortcut learner, with no error.
- Abstraction and exemplar matching differ only in held-out transfer.

**Results.**
- **Pretrained models.** Nine pretrained LMs from three families were tested: Qwen3 0.6–8B (base and post-trained), SmolLM2-1.7B and OLMo-2 1B/7B. All are recall-only. None infers sparse-cue Latin constraints in direct-answer mode, and none compounds: t=4 minus length-matched scramble is −0.02 to +0.03, with every CI including 0. Recall of a previously seen labelled table, by contrast, reaches 0.95–0.96 in Qwen3 at 4–8B.
- **Trained networks.** Small transformers trained from scratch on group, Latin and loop diets converge instead to a global counting shortcut. They answer with the symbol least represented in the cue: five of the six learned networks match an exact count learner to within 0.001–0.007, and the 6-layer network exceeds it by only 0.014. The fingerprint classifies them accordingly.
- **Positive control.** The released In-Context Algebra model, a transformer trained on in-context group tables, is classified above recall-only. Latin- and loop-forced cells score 0.89–0.99, and associativity-only cells 0.53–0.70, so the group structure it has learned is partial. In an exploratory check of six released training checkpoints, its class changes between 5k and 10k steps.
- **CR axis.** We also test a published activation-space "compute vs retrieve" axis on the format-matched leak/base contrast. It does not track retrieval vs inference: it is supported in 0 of 5 models and significantly reversed in 3. An exploratory vocabulary-overlap control attributes 64–87% of the reversal to context vocabulary. With vocabulary matched, a small retrieval signal of the predicted sign emerges in the post-trained model.

**Contribution.** We release:
- a measurement whose classes are exact and preregistered;
- a calibrated null for direct-answer in-context algebra up to 8B;
- a count baseline that any claim of in-context constraint inference should beat;
- a cheap matched-format test for activation-space "mode" probes.

All model inference and training used a single 8 GB laptop GPU.
