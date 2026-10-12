"""Post-hoc type-matched check of the held-out contrast (added 2026-10-12, after an adversarial review; not preregistered).

Held-out targets are non-abelian types only (n=6: S3; n=8: D4 or Q8) with abelian histories, while the base and scramble
baselines that the held-out cell is compared with mix target types. This script restricts the base t=0, base t=4 and
scramble t=4 cells to trials whose target type is one of the held-out target types, and recomputes pooled G-cell accuracy
and the contrasts
    held4 - scr4   (type-matched; the published raw held4 - scr4 uses all scramble trials)
    held4 - t0     (type-matched)
    t4 - held4     (same target types, with versus without the target's type in the history: the exemplar advantage)
with trial-bootstrap 95% CIs (2,000 resamples per contrast, each cell resampled independently, seeded from the data).

usage: python heldout_typematched.py   -> writes heldout_typematched.md and heldout_typematched.json
"""
import json, os, sys, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect import ROOT, ROWS  # noqa: E402

NBOOT = 2000
CONDS = ['base_t0', 'base_t4', 'scramble_t4', 'heldout_t4']


def load(path):
    """-> {cell: [(target_type, correct G, number of G cells), ...] per trial}"""
    cells = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        g = [q['correct'] for q in r['queries'] if q['class'] == 'G']
        cells.setdefault(r['file'].replace('.jsonl', ''), []).append(('/'.join(map(str, r['target_type'])), sum(g), len(g)))
    return cells


def pooled(trials, idx=None):
    idx = range(len(trials)) if idx is None else idx
    c = sum(trials[i][1] for i in idx); t = sum(trials[i][2] for i in idx)
    return c / t if t else float('nan')


def analyse(cells, n, seed):
    held = cells[f'n{n}_heldout_t4']
    types = sorted({tt for tt, _, _ in held})
    sub = {c: [x for x in cells[f'n{n}_{c}'] if x[0] in types] for c in CONDS[:3]}
    sub['heldout_t4'] = held
    allscr = cells[f'n{n}_scramble_t4']
    G = {c: pooled(v) for c, v in sub.items()}
    est = {'held_minus_scr': G['heldout_t4'] - G['scramble_t4'], 'held_minus_t0': G['heldout_t4'] - G['base_t0'],
           't4_minus_held': G['base_t4'] - G['heldout_t4'], 'held_minus_scr_alltypes': G['heldout_t4'] - pooled(allscr)}
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in est}
    for _ in range(NBOOT):
        g = {c: pooled(v, rng.integers(0, len(v), len(v))) for c, v in sub.items()}
        ga = pooled(allscr, rng.integers(0, len(allscr), len(allscr)))
        boot['held_minus_scr'].append(g['heldout_t4'] - g['scramble_t4'])
        boot['held_minus_t0'].append(g['heldout_t4'] - g['base_t0'])
        boot['t4_minus_held'].append(g['base_t4'] - g['heldout_t4'])
        boot['held_minus_scr_alltypes'].append(g['heldout_t4'] - ga)
    ci = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in boot.items()}
    return {'types': types, 'trials': {c: len(v) for c, v in sub.items()}, 'G': G, 'est': est, 'ci': ci}


def fmt(x, ci):
    return f'{x:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]'


def main():
    out, md = {}, []
    md.append('# Type-matched held-out check (post hoc, not preregistered)\n')
    md.append('Base and scramble cells restricted to trials whose target type is a held-out target type '
              '(n=6: S3; n=8: D4 or Q8). Pooled G-cell accuracy; trial-bootstrap 95% CIs, 2,000 resamples. '
              '"held - scr (all)" is the published raw contrast against all scramble trials, for comparison.\n')
    for n in (6, 8):
        md.append(f'\n## n={n}\n')
        md.append('| group | learner/model | precision | trials t0/t4/scr/held | G t0 | G t4 | G scr4 | G held4 | '
                  'held - scr (type-matched) | held - t0 (type-matched) | t4 - held (type-matched) | held - scr (all) |')
        md.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for group, label, path, prec in ROWS:
            cells = load(os.path.join(ROOT, path))
            if f'n{n}_heldout_t4' not in cells:
                continue
            r = analyse(cells, n, zlib.crc32(f'{label}|{path}|{n}'.encode()))
            out.setdefault(f'{label} ({prec})', {})[f'n{n}'] = r
            tr, G, e, c = r['trials'], r['G'], r['est'], r['ci']
            md.append(f"| {group} | {label} | {prec} | {tr['base_t0']}/{tr['base_t4']}/{tr['scramble_t4']}/{tr['heldout_t4']} | "
                      f"{G['base_t0']:.3f} | {G['base_t4']:.3f} | {G['scramble_t4']:.3f} | {G['heldout_t4']:.3f} | "
                      f"{fmt(e['held_minus_scr'], c['held_minus_scr'])} | {fmt(e['held_minus_t0'], c['held_minus_t0'])} | "
                      f"{fmt(e['t4_minus_held'], c['t4_minus_held'])} | {fmt(e['held_minus_scr_alltypes'], c['held_minus_scr_alltypes'])} |")
        md.append(f"\nHeld-out target types at n={n}: {', '.join(t.split('/')[-1] for t in r['types'])}.")
    open(os.path.join(HERE, 'heldout_typematched.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    json.dump(out, open(os.path.join(HERE, 'heldout_typematched.json'), 'w', encoding='utf-8'), indent=1)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
