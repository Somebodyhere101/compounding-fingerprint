# Released In-Context Algebra checkpoints

Experiment 6 scores the transformer released by Todd, Brinkmann, Gandikota and Bau, *In-Context Algebra* (ICLR 2026; arXiv 2512.16902).

- `weights/` (not versioned) holds the seven checkpoints `algebra_gpt_{0,5000,10000,20000,40000,100000,best}.pt`, downloaded from <https://algebra.baulab.info/weights/mixrosette-facts-10-16-8heads/models/>. Check them with `cd weights && sha256sum -c ../weights_SHA256SUMS`.
- `metadata.json` is the training configuration published with those checkpoints on the authors' weights page (<https://algebra.baulab.info/weights/>). It is kept unchanged as provenance for the model and token figures in `../RESULTS.md`. All rights remain with its authors.
- `upstream/` (not versioned) was a local clone of <https://github.com/ericwtodd/algebra> at commit 532a6a2 (MIT License). It was read, not executed; `../ica.py` re-implements the model.
