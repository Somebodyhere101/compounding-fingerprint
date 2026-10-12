# G-cell accuracy by target structure and transpose availability (post hoc; ica_best_main, stimulus order; pooled over the base t0/t1 and heldout t1 cells of each order)

Baselines are expected accuracies on the same cells: ideal N = uniform over Latin-feasible values; loop level = uniform over values a loop (Latin square with two-sided identity) completion allows (res/loop_level.jsonl, loop_level.py); ideal C = count shortcut.

| order | target | transpose in cue | transpose vs gold | G acc | n | pred = revealed transpose | ideal N (Latin only) | loop level (exact SAT) | ideal C (count) |
|---|---|---|---|---|---|---|---|---|---|
| n5 | Z5 |  |  | 0.531 | 533 | 0.131 | 0.320 | 0.459 | 0.256 |
| n5 | abelian | transpose hidden |  | 0.460 | 463 | 0.000 | 0.322 | 0.456 | 0.290 |
| n5 | abelian | transpose revealed | transpose = gold | 1.000 | 70 | 1.000 | 0.304 | 0.474 | 0.029 |
| n6 | S3 |  |  | 0.561 | 2511 | 0.100 | 0.298 | 0.419 | 0.236 |
| n6 | Z6 |  |  | 0.694 | 1086 | 0.190 | 0.296 | 0.427 | 0.231 |
| n6 | abelian | transpose hidden |  | 0.628 | 872 | 0.000 | 0.300 | 0.430 | 0.272 |
| n6 | abelian | transpose revealed | transpose = gold | 0.963 | 214 | 0.963 | 0.281 | 0.415 | 0.063 |
| n6 | non-abelian | transpose hidden |  | 0.542 | 1947 | 0.000 | 0.301 | 0.417 | 0.239 |
| n6 | non-abelian | transpose revealed | transpose != gold | 0.583 | 504 | 0.377 | 0.285 | 0.427 | 0.248 |
| n6 | non-abelian | transpose revealed | transpose = gold | 1.000 | 60 | 1.000 | 0.295 | 0.417 | 0.057 |
| n8 | D4 |  |  | 0.601 | 2941 | 0.133 | 0.220 | 0.290 | 0.160 |
| n8 | Q8 |  |  | 0.475 | 3338 | 0.157 | 0.218 | 0.287 | 0.156 |
| n8 | Z2^3 |  |  | 0.868 | 1074 | 0.205 | 0.223 | 0.301 | 0.163 |
| n8 | Z4xZ2 |  |  | 0.646 | 1315 | 0.183 | 0.219 | 0.281 | 0.157 |
| n8 | Z8 |  |  | 0.578 | 1676 | 0.200 | 0.218 | 0.292 | 0.159 |
| n8 | abelian | transpose hidden |  | 0.610 | 3206 | 0.000 | 0.221 | 0.292 | 0.194 |
| n8 | abelian | transpose revealed | transpose = gold | 0.925 | 859 | 0.925 | 0.212 | 0.286 | 0.032 |
| n8 | non-abelian | transpose hidden |  | 0.501 | 4918 | 0.000 | 0.220 | 0.289 | 0.171 |
| n8 | non-abelian | transpose revealed | transpose != gold | 0.425 | 760 | 0.458 | 0.216 | 0.290 | 0.168 |
| n8 | non-abelian | transpose revealed | transpose = gold | 0.943 | 601 | 0.943 | 0.215 | 0.283 | 0.038 |

Model minus loop level on transpose-hidden G cells (trial bootstrap, 2,000 resamples, seed fixed per order):

- n5: +0.004 [-0.043, +0.048] (53 trials, 463 cells)
- n6: +0.148 [+0.130, +0.166] (282 trials, 2819 cells)
- n8: +0.254 [+0.236, +0.273] (300 trials, 8124 cells)
