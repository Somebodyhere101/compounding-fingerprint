"""Builds the report tables from ../out/sim_n*.json (env-averaged over target iso classes, uniform weights).

usage: python report_tables.py [theta=0.9]
prints markdown tables:
  T1  k*_theta(t), t=0..4, baseline condition, per learner and n
  T2  composition of unseen cells by forcing class at design k
  T3  accuracy on G-cells (group-forced, not loop-forced) and O-cells (loop-forced, not Latin-forced) at design k, t=0..4
  T4  held-out iso class (n=4: V4 after Z4-only; n=6: S3 after Z6-only): k* and G/O-cell accuracy
  T5  wrong abstraction (n=5): group target after non-associative loop history; loop target after group history
  T6  lesion predictions: P(full self-completion correct) at k1 and implied stage-2 accuracy
"""
import sys, json, os
import numpy as np

OUT = '../out'
LEARN = ['R', 'Np', 'N', 'Nb', 'Niso', 'Nisob', 'A', 'Ab', 'Asl', 'Bposs', 'B', 'Bh']
DESIGN_K = {4: 4, 5: 6, 6: 10, 7: 12, 8: 16}


def load(n):
    p = f'{OUT}/sim_n{n}.json'
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    d['acc'] = {tuple(k.split('|')): v for k, v in d['acc'].items()}
    d['cnt'] = {tuple(k.split('|')): v for k, v in d['cnt'].items()}
    d['pfull'] = {tuple(k.split('|')): v for k, v in d['pfull'].items()}
    return d


def targets(d, cond):
    return sorted({k[1] for k in d['acc'] if k[0] == cond})


def env_curve(d, cond, L, t, cc='all', tg=None):
    """accuracy vs k averaged over targets (uniform), weighting cells by their counts within class cc."""
    n = d['n']
    tgs = tg if tg is not None else [x for x in targets(d, cond) if not x.startswith('lp')]
    ks = [k for k in d['ks'] if k < n * n]
    out = []
    for k in ks:
        num = den = 0.0
        for x in tgs:
            a = d['acc'].get((cond, x, L, str(t), str(k), cc))
            c = d['cnt'].get((cond, x, str(t), str(k), cc), 0.0)
            if a is None or c == 0:
                continue
            num += a * c; den += c
        out.append(num / den if den > 0 else np.nan)
    return ks, out


def kstar(d, cond, L, t, theta, tg=None):
    ks, c = env_curve(d, cond, L, t, 'all', tg)
    for k, a in zip(ks, c):
        if not np.isnan(a) and a >= theta - 1e-12:
            return k
    return None


def fmt(v):
    return '-' if v is None or (isinstance(v, float) and np.isnan(v)) else (f'{v:.2f}' if isinstance(v, float) else str(v))


def main(theta=0.9):
    print(f'## T1 baseline k*_{theta}(t), t=0..4 (env-averaged over iso classes)\n')
    print('| n | ' + ' | '.join(LEARN) + ' |')
    print('|' + '---|' * (len(LEARN) + 1))
    for n in (4, 5, 6, 7, 8):
        d = load(n)
        if d is None:
            continue
        row = []
        for L in LEARN:
            ks = [kstar(d, 'base', L, t, theta) for t in range(5)]
            if all(k is None for k in ks) and L != 'R':
                row.append('n/a')
            else:
                row.append(','.join(fmt(k) for k in ks))
        print(f'| {n} | ' + ' | '.join(row) + ' |')
    print('\n## T2 composition of unseen cells (mean count per trial) at design k, baseline, env-averaged\n')
    print('| n | k | unseen | L | O | G | Ab | Cy | U |')
    print('|---|---|---|---|---|---|---|---|---|')
    for n in (4, 5, 6, 7, 8):
        d = load(n)
        if d is None:
            continue
        kd = DESIGN_K[n]
        tgs = [x for x in targets(d, 'base') if not x.startswith('lp')]
        vals = []
        for cc in ['all', 'L', 'O', 'G', 'Ab', 'Cy', 'U']:
            vals.append(np.mean([d['cnt'].get(('base', x, '0', str(kd), cc), 0.0) for x in tgs]))
        print(f'| {n} | {kd} | ' + ' | '.join(f'{v:.2f}' for v in vals) + ' |')
    print('\n## T3 accuracy at design k on O-cells (loop-forced, not Latin) and G-cells (group-forced, not loop), t=0..4\n')
    print('| n | k | class | ' + ' | '.join(LEARN) + ' |')
    print('|' + '---|' * (len(LEARN) + 3))
    for n in (4, 5, 6, 7, 8):
        d = load(n)
        if d is None:
            continue
        kd = DESIGN_K[n]
        for cc in ('O', 'G', 'all'):
            row = []
            for L in LEARN:
                vs = []
                for t in range(5):
                    ks, c = env_curve(d, 'base', L, t, cc)
                    vs.append(c[ks.index(kd)])
                row.append(','.join(fmt(v) for v in vs))
            print(f'| {n} | {kd} | {cc} | ' + ' | '.join(row) + ' |')
    print('\n## T4 held-out iso class (history from seen class only)\n')
    print('| n | target | measure | ' + ' | '.join(LEARN) + ' |')
    print('|' + '---|' * (len(LEARN) + 3))
    for n in (4, 6, 8):
        d = load(n)
        if d is None:
            continue
        kd = DESIGN_K[n]
        for tg in targets(d, 'heldout'):
            row = [','.join(fmt(kstar(d, 'heldout', L, t, theta, [tg])) for t in range(5)) for L in LEARN]
            print(f'| {n} | {tg} | k*(t) | ' + ' | '.join(row) + ' |')
            for cc in ('O', 'G'):
                row = []
                for L in LEARN:
                    vs = []
                    for t in range(5):
                        ks, c = env_curve(d, 'heldout', L, t, cc, [tg])
                        vs.append(c[ks.index(kd)])
                    row.append(','.join(fmt(v) for v in vs))
                print(f'| {n} | {tg} | acc {cc}@k={kd} | ' + ' | '.join(row) + ' |')
    d = load(5)
    if d is not None:
        print('\n## T5 wrong abstraction, n=5 (history = t non-associative loops, uniform over the 5 iso classes)\n')
        print('| target | history | measure | ' + ' | '.join(LEARN) + ' |')
        print('|' + '---|' * (len(LEARN) + 3))
        kd = DESIGN_K[5]
        lps = [x for x in targets(d, 'loops') if x.startswith('lp')]
        for tgset, tname in (([('Z5')], 'Z5'), (lps, 'loops(avg)')):
            for cond in ('loops', 'base'):
                row = [','.join(fmt(kstar(d, cond, L, t, theta, tgset)) for t in range(5)) for L in LEARN]
                print(f'| {tname} | {cond} | k*(t) | ' + ' | '.join(row) + ' |')
                for cc in (('G', 'O', 'all') if tname == 'Z5' else ('O', 'U', 'all')):
                    row = []
                    for L in LEARN:
                        vs = []
                        for t in range(5):
                            ks, c = env_curve(d, cond, L, t, cc, tgset)
                            vs.append(c[ks.index(kd)])
                        row.append(','.join(fmt(v) for v in vs))
                    print(f'| {tname} | {cond} | acc {cc}@k={kd} | ' + ' | '.join(row) + ' |')
    print('\n## T6 lesion: P(learner completes a stage-1 target fully correctly) at k1, history t=4; implied stage-2 G-cell accuracy at design k\n')
    print('| n | k1 | ' + ' | '.join(LEARN) + ' |')
    print('|' + '---|' * (len(LEARN) + 2))
    for n in (4, 5, 6, 7, 8):
        d = load(n)
        if d is None:
            continue
        tgs = [x for x in targets(d, 'base') if not x.startswith('lp')]
        for k1 in sorted({DESIGN_K[n], DESIGN_K[n] + 4}):
            row = []
            for L in LEARN:
                ps = [d['pfull'].get(('base', x, L, '4', str(k1))) for x in tgs]
                ps = [p for p in ps if p is not None]
                if not ps:
                    row.append('-'); continue
                P1 = float(np.mean(ps))
                kd = DESIGN_K[n]
                ks, c1 = env_curve(d, 'base', L, 1, 'G' if n >= 5 else 'O')
                ks, c0 = env_curve(d, 'base', L, 0, 'G' if n >= 5 else 'O')
                a1, a0 = c1[ks.index(kd)], c0[ks.index(kd)]
                row.append(f'P1={P1:.2f}; acc={P1 * a1 + (1 - P1) * a0:.2f}')
            print(f'| {n} | {k1} | ' + ' | '.join(row) + ' |')


if __name__ == '__main__':
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.9)
