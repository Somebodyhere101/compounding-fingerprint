"""Summaries of sim_n*.json: k*(t) curves, per-class accuracies, cell-class composition.

usage: python summarize.py ../out/sim_n5.json [theta=0.9] [kdesign]
"""
import sys, json
import numpy as np

LEARNERS = ['R', 'Np', 'N', 'Nb', 'Niso', 'Nisob', 'A', 'Ab', 'Asl', 'Bposs', 'B', 'Bh']
CC = ['L', 'O', 'G', 'Ab', 'Cy', 'U']


def load(path):
    d = json.load(open(path))
    acc = {tuple(k.split('|')): v for k, v in d['acc'].items()}
    cnt = {tuple(k.split('|')): v for k, v in d['cnt'].items()}
    pfull = {tuple(k.split('|')): v for k, v in d['pfull'].items()}
    wgrp = {tuple(k.split('|')): v for k, v in d['wgrp'].items()}
    return d, acc, cnt, pfull, wgrp


def curve(acc, cond, tgt, L, t, ks, cc='all'):
    return [acc.get((cond, tgt, L, str(t), str(k), cc), np.nan) for k in ks]


def kstar(acc, cond, tgt, L, t, ks, theta):
    c = curve(acc, cond, tgt, L, t, ks)
    for k, a in zip(ks, c):
        if not np.isnan(a) and a >= theta:
            return k
    return None


def main(path, theta=0.9, kdesign=None):
    d, acc, cnt, pfull, wgrp = load(path)
    n = d['n']
    ks = [k for k in d['ks'] if k < n * n]
    conds = sorted({(k[0], k[1]) for k in acc})
    print(f'== {path}  n={n}  M={d["M"]} cues per (target,k)  theta={theta}')
    for cond, tgt in conds:
        print(f'\n-- condition={cond} target={tgt}:  k*_{theta}(t) for t=0..4')
        for L in LEARNERS:
            row = [kstar(acc, cond, tgt, L, t, ks, theta) for t in range(5)]
            if all(r is None for r in row) and (cond, tgt, L, '0', '0', 'all') not in acc:
                continue
            print(f'   {L:6s}', row)
        # composition of unseen cells
    print('\n-- composition: mean number of unseen cells per class (cond=base)')
    tg = sorted({k[1] for k in cnt if k[0] == 'base'})
    for tgt in tg:
        print(f'   target {tgt}:')
        for k in ks:
            comp = {c: cnt.get(('base', tgt, '0', str(k), c), 0.0) for c in CC}
            tot = cnt.get(('base', tgt, '0', str(k), 'all'), 0.0)
            print(f'     k={k:3d} unseen={tot:5.1f} ' + ' '.join(f'{c}={comp[c]:5.2f}' for c in CC))
    if kdesign is not None:
        print(f'\n-- accuracy per cell class at k={kdesign}, t=0..4')
        for cond, tgt in conds:
            print(f'   condition={cond} target={tgt}')
            for L in LEARNERS:
                parts = []
                for cc in ['all'] + CC:
                    vals = [acc.get((cond, tgt, L, str(t), str(kdesign), cc), np.nan) for t in range(5)]
                    if all(np.isnan(v) for v in vals):
                        continue
                    parts.append(f'{cc}:' + ','.join(f'{v:.2f}' for v in vals))
                if parts:
                    print(f'     {L:6s} ' + ' | '.join(parts))


if __name__ == '__main__':
    path = sys.argv[1]
    theta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
    kd = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(path, theta, kd)
