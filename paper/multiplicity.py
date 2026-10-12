"""Post-hoc multiple-comparisons check (added 2026-10-12, after an adversarial review; not preregistered).

The paper reports uncorrected trial-bootstrap 95% CIs. This script recomputes, for the ten pretrained-model
configurations, the raw G-accuracy contrasts t4 - t0 (20 tests: 10 configurations x n=6, n=8) with Bonferroni-adjusted
intervals (1 - 0.05/20 = 99.75%), and the primary n=6 null contrasts t4 - scramble (10 tests) with 99.5% intervals,
whose upper bounds are what the null claims rest on. 20,000 resamples per contrast, each cell resampled independently,
seeded from the data.

usage: python multiplicity.py   -> writes multiplicity.md and multiplicity.json
"""
import json, os, sys, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect import ROOT, ROWS  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, 'exp1'))
import analyze as A1  # noqa: E402

NBOOT = 20000


def contrast(cells, a, b, seed, levels):
    ta, tb = cells[a]['trials'], cells[b]['trials']
    est = A1.pooled(ta, 'G')[0] - A1.pooled(tb, 'G')[0]
    rng = np.random.default_rng(seed)
    v = np.empty(NBOOT)
    for i in range(NBOOT):
        v[i] = A1.pooled(ta, 'G', rng.integers(0, len(ta), len(ta)))[0] - A1.pooled(tb, 'G', rng.integers(0, len(tb), len(tb)))[0]
    p = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))   # two-sided bootstrap p-value
    return est, {lv: (float(np.percentile(v, 50 - lv / 2)), float(np.percentile(v, 50 + lv / 2))) for lv in levels}, p


def fmt(e, ci):
    return f'{e:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]'


def main():
    rows, out = [], {}
    for group, label, path, prec in ROWS:
        if group != 'llm':
            continue
        cells = A1.load(os.path.join(ROOT, path))
        name = f'{label} ({prec})'
        out[name] = {}
        for n in (6, 8):
            s = zlib.crc32(f'{name}|{n}'.encode())
            e, ci, p = contrast(cells, f'n{n}_base_t4', f'n{n}_base_t0', s, (95, 99.75))
            out[name][f'n{n}_t4_t0'] = {'est': e, 'ci95': ci[95], 'ci9975': ci[99.75], 'p': p}
        e, ci, p = contrast(cells, 'n6_base_t4', 'n6_scramble_t4', zlib.crc32(f'{name}|scr'.encode()), (95, 99.5))
        out[name]['n6_t4_scr'] = {'est': e, 'ci95': ci[95], 'ci995': ci[99.5], 'p': p}
    md = ['# Multiple-comparisons check (post hoc, not preregistered)\n',
          'Raw G-accuracy contrasts for the ten pretrained-model configurations; trial bootstrap, 20,000 resamples. '
          'Bonferroni: 20 tests for t4 - t0 (99.75% intervals, p < 0.0025), 10 tests for the n=6 t4 - scramble nulls (99.5% intervals).\n',
          '| model | n=6 t4 - t0 [95%] | [99.75%] | p | n=8 t4 - t0 [95%] | [99.75%] | p | n=6 t4 - scr [95%] | [99.5%] |',
          '|---|---|---|---|---|---|---|---|---|']
    for name, d in out.items():
        a, b, c = d['n6_t4_t0'], d['n8_t4_t0'], d['n6_t4_scr']
        md.append(f"| {name} | {fmt(a['est'], a['ci95'])} | [{a['ci9975'][0]:+.3f}, {a['ci9975'][1]:+.3f}] | {a['p']:.4f} | "
                  f"{fmt(b['est'], b['ci95'])} | [{b['ci9975'][0]:+.3f}, {b['ci9975'][1]:+.3f}] | {b['p']:.4f} | "
                  f"{fmt(c['est'], c['ci95'])} | [{c['ci995'][0]:+.3f}, {c['ci995'][1]:+.3f}] |")
    ps = [(name, k, d[k]['p'], d[k]['est']) for name, d in out.items() for k in ('n6_t4_t0', 'n8_t4_t0')]
    sig95 = [x for x in ps if x[2] < 0.05]
    sigb = [x for x in ps if x[2] < 0.05 / 20]
    md.append(f'\nt4 - t0: {len(sig95)} of 20 contrasts have p < 0.05 '
              f"({sum(1 for x in sig95 if x[3] > 0)} increases, {sum(1 for x in sig95 if x[3] < 0)} decreases); "
              f'{len(sigb)} survive Bonferroni (p < 0.0025): ' + ', '.join(f'{x[0]} {x[1][:2]} {x[3]:+.3f}' for x in sigb) + '.')
    ub = max(d['n6_t4_scr']['ci995'][1] for d in out.values())
    md.append(f'Largest Bonferroni-adjusted (99.5%) upper bound of the n=6 t4 - scramble nulls: {ub:+.3f}.')
    open(os.path.join(HERE, 'multiplicity.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    json.dump(out, open(os.path.join(HERE, 'multiplicity.json'), 'w', encoding='utf-8'), indent=1)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
