"""Summaries for abc_n6.json (curriculum A->B->C) and inweights_n5.json (prior in weights).
usage: python summ_abc_w.py
"""
import json, os
import numpy as np


def kstar(curve, theta=0.9):
    for k, a in curve:
        if a is not None and a >= theta - 1e-12:
            return k
    return None


if os.path.exists('../out/abc_n6.json'):
    d = json.load(open('../out/abc_n6.json'))
    acc = {tuple(k.split('|')): v for k, v in d['acc'].items()}
    names = sorted({k[0] for k in acc})
    Ls = ['N', 'Niso', 'A_cum', 'A_rec', 'Bh', 'Bh_cp']
    ks = sorted({int(k[2]) for k in acc})
    print('## (iii) curriculum n=6: history = 2 tables from {Z6,S3} then 2 Z6 tables; target C')
    for grp, label in ((['S3'], 'S3 (non-abelian group)'), ([x for x in names if x.startswith('loop')], 'group-like non-assoc loops (avg of 5)')):
        print(f'target {label}')
        for L in Ls:
            curve = []
            for k in ks:
                vals = [acc.get((x, L, str(k), 'all')) for x in grp]
                vals = [v for v in vals if v is not None]
                curve.append((k, float(np.mean(vals)) if vals else None))
            row = {cc: np.mean([acc.get((x, L, '10', cc), np.nan) for x in grp]) for cc in ('O', 'G', 'U', 'all')}
            print(f'   {L:6s} k*_0.9={kstar(curve)}   acc@k=10: ' + ' '.join(f'{cc}={v:.2f}' for cc, v in row.items()))

if os.path.exists('../out/inweights_n5.json'):
    d = json.load(open('../out/inweights_n5.json'))
    acc = {tuple(k.split('|')): v for k, v in d['acc'].items()}
    names = sorted({k[0] for k in acc})
    print('\n## (iv) prior in weights, n=5: A_w rigid group prior; Bh_w hierarchical, hyperprior favours groups 2^10:1')
    for grp, label in ((['Z5'], 'Z5'), ([x for x in names if x.startswith('lp')], 'loops (avg of 5 iso classes)')):
        for hk in ('groups', 'loops'):
            for L in ('A_w', 'Bh_w'):
                ks_t = []
                accs = []
                for t in range(5):
                    curve = []
                    for k in range(25):
                        vals = [acc.get((x, hk, L, str(t), str(k), 'all')) for x in grp]
                        vals = [v for v in vals if v is not None]
                        curve.append((k, float(np.mean(vals)) if vals else None))
                    ks_t.append(kstar(curve))
                    accs.append(np.mean([acc.get((x, hk, L, str(t), '6', 'all'), np.nan) for x in grp]))
                print(f'   target {label:28s} history={hk:6s} {L:5s} k*_0.9(t=0..4)={ks_t}  acc_all@k=6(t=0..4)=' + ','.join(f'{a:.2f}' for a in accs))
