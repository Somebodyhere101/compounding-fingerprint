"""Independent DFS check for n=9: no defining cue of size <=4 under the group prior;
and the ILP's 5-cues are defining.  usage: python check9.py"""
import time, json
import numpy as np
from tables import labelled_groups, groups_of_order
from defsets import min_defining, bitsets_for
G, names = labelled_groups(9)
ilp = json.load(open('../out/md_ilp_9.json'))
for gname, T in groups_of_order(9).items():
    t0 = time.time()
    k, cnt, ex = min_defining(T, G, kmax=4, count_all=False)
    print(gname, 'DFS: defining cue of size<=4 found?', k, ex, f'({time.time()-t0:.0f}s)', flush=True)
    cue = ilp[f'9:{gname}:group']['cue']
    m = np.ones(len(G), bool)
    for (i, j, v) in cue:
        m &= G[:, i, j] == v
    print(gname, 'ILP 5-cue consistent tables:', int(m.sum()), flush=True)
