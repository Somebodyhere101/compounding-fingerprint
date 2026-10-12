"""Independent check of the order-10 minimum cues: count labelled group tables of order 10 consistent with
each 5-cell cue (must be exactly 1), and confirm the cue values against the canonical tables.
usage: python check10.py"""
import json
import numpy as np
from tables import cyclic, closure, from_perms
from md_extra import labelled_coset_reps
D5 = from_perms(closure([(1, 2, 3, 4, 0), (0, 4, 3, 2, 1)]))
gs = {'Z10': cyclic(10), 'D5': D5}
C = np.concatenate([labelled_coset_reps(T)[0] for T in gs.values()])
assert len(C) == 1088640 and len({x.tobytes() for x in C[::997]}) == len(C[::997])
res = json.load(open('../out/md_n10.json'))
for h, T in gs.items():
    cue = res[h]['cue']
    assert all(int(T[a, b]) == v for a, b, v in cue)
    m = np.ones(len(C), bool)
    for a, b, v in cue:
        m &= C[:, a, b] == v
    print(h, 'cue', cue, '-> consistent labelled group tables:', int(m.sum()), '| equals target:', bool(np.array_equal(C[m][0], T)))
