"""Analyse Experiment 1 results with the preregistered decision rule (PREREGISTRATION.md).

usage: python analyze.py            -> prints tables, writes RESULTS_tables.md and results_summary.json
"""
import glob, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ['L', 'O', 'G', 'Ab', 'Cy', 'U']
RNG = np.random.default_rng(12345)  # legacy; decision() now seeds its own generator from the data (audit fix 2026-10-07)


def _seed(cells, prefix, field):
    import zlib
    key = prefix + str(field) + ''.join(f"{c}:{len(cells[c]['trials'])}:{sum(v[0] for t in cells[c]['trials'] for v in t.values())}" for c in sorted(cells) if c.startswith(prefix))
    return zlib.crc32(key.encode())
NBOOT = 2000


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load(path):
    """-> {cell: {'n':, 'trials': [ {class: (correct, total, sum_pgold)} ]}}"""
    cells = {}
    for l in open(path, encoding='utf-8'):
        r = json.loads(l)
        cell = r['file'].replace('.jsonl', '')
        d = cells.setdefault(cell, {'n': r['n'], 'trials': []})
        per = {}
        for q in r['queries']:
            c, t, s = per.get(q['class'], (0, 0, 0.0))
            per[q['class']] = (c + q['correct'], t + 1, s + q['p_gold'])
        d['trials'].append(per)
    return cells


def pooled(trials, cls, idx=None, field=0):
    """pooled accuracy (field 0) or mean p_gold (field 2) over the given trial indices"""
    idx = range(len(trials)) if idx is None else idx
    num = 0.0; den = 0
    for i in idx:
        v = trials[i].get(cls)
        if v:
            num += v[field]; den += v[1]
    return (num / den if den else float('nan')), den


def decision(cells, n, prefix, field=0, boot=True):
    """Preregistered rule on cells with the given prefix (e.g. 'n6')."""
    need = [f'{prefix}_base_t0', f'{prefix}_base_t1', f'{prefix}_base_t4', f'{prefix}_scramble_t4', f'{prefix}_heldout_t4']
    if any(c not in cells for c in need):
        return None
    ch = 1.0 / n

    def stats(sel):
        accL, _ = pooled(cells[f'{prefix}_base_t0']['trials'], 'L', sel[f'{prefix}_base_t0'], field)
        G = {c: pooled(cells[c]['trials'], 'G', sel[c], field)[0] for c in need}
        den = accL - ch
        g = {c: (G[c] - ch) / den if den > 1e-9 else float('nan') for c in need}
        base = max(g[f'{prefix}_base_t0'], g[f'{prefix}_scramble_t4'])
        comp = g[f'{prefix}_base_t4'] - base
        held = g[f'{prefix}_heldout_t4'] - base
        w = held / comp if abs(comp) > 1e-9 else float('nan')
        shape = g[f'{prefix}_base_t4'] - g[f'{prefix}_base_t1']
        return dict(accL=accL, G=G, g=g, base=base, comp=comp, held=held, w=w, shape=shape)

    full = {c: list(range(len(cells[c]['trials']))) for c in need}
    s = stats(full)
    # classification
    if s['accL'] < ch + 0.25:
        cls = 'R'
    elif s['g'][f'{prefix}_base_t0'] >= 0.85:
        cls = 'B-type'
    elif s['comp'] < 0.15:
        cls = 'N'
    elif s['held'] >= 0.6 * s['comp']:
        cls = 'A'
    else:
        cls = 'N_iso'
    s['class'] = cls
    if boot:
        RNG = np.random.default_rng(_seed(cells, prefix, field))
        bs = {k: [] for k in ['comp', 'held', 'shape', 'accL']}
        for _ in range(NBOOT):
            sel = {c: RNG.integers(0, len(cells[c]['trials']), len(cells[c]['trials'])) for c in need}
            b = stats(sel)
            for k in bs:
                bs[k].append(b[k])
        s['ci'] = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in bs.items()}
        # raw (unscaled) G-accuracy contrasts, robust to a noisy acc_L
        rawb = {'dG_t4_t0': [], 'dG_held_t0': [], 'dG_t4_scr': []}
        for _ in range(NBOOT):
            sel = {c: RNG.integers(0, len(cells[c]['trials']), len(cells[c]['trials'])) for c in need}
            G = {c: pooled(cells[c]['trials'], 'G', sel[c], field)[0] for c in need}
            rawb['dG_t4_t0'].append(G[f'{prefix}_base_t4'] - G[f'{prefix}_base_t0'])
            rawb['dG_held_t0'].append(G[f'{prefix}_heldout_t4'] - G[f'{prefix}_base_t0'])
            rawb['dG_t4_scr'].append(G[f'{prefix}_base_t4'] - G[f'{prefix}_scramble_t4'])
        G = s['G']
        s['raw'] = {'dG_t4_t0': G[f'{prefix}_base_t4'] - G[f'{prefix}_base_t0'],
                    'dG_held_t0': G[f'{prefix}_heldout_t4'] - G[f'{prefix}_base_t0'],
                    'dG_t4_scr': G[f'{prefix}_base_t4'] - G[f'{prefix}_scramble_t4']}
        s['raw_ci'] = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in rawb.items()}
    return s


def main():
    out_md = []
    summary = {}
    for path in sorted(glob.glob(os.path.join(HERE, 'res', '*.jsonl'))):
        model = os.path.basename(path).replace('.jsonl', '')
        if model.startswith('_'):
            continue
        cells = load(path)
        summary[model] = {'cells': {}}
        out_md.append(f'\n## {model}\n')
        out_md.append('| cell | trials | chance | L | O | G | U | G mean p_gold | all |')
        out_md.append('|---|---|---|---|---|---|---|---|---|')
        for cell in sorted(cells, key=lambda c: (int(c.split('_')[0][1:]), c)):
            d = cells[cell]; tr = d['trials']; n = d['n']
            row = [cell, str(len(tr)), f'{1 / n:.2f}']
            cs = {}
            for cl in ['L', 'O', 'G', 'U']:
                a, m = pooled(tr, cl)
                k = sum(t[cl][0] for t in tr if cl in t)
                lo, hi = wilson(k, m)
                cs[cl] = {'acc': a, 'n': m, 'ci': (lo, hi)}
                row.append('—' if m == 0 else f'{a:.3f} [{lo:.2f},{hi:.2f}] (n={m})')
            pg, _ = pooled(tr, 'G', field=2)
            row.append('—' if math.isnan(pg) else f'{pg:.3f}')
            tot_c = sum(v[0] for t in tr for v in t.values()); tot_n = sum(v[1] for t in tr for v in t.values())
            row.append(f'{tot_c / tot_n:.3f}')
            out_md.append('| ' + ' | '.join(row) + ' |')
            summary[model]['cells'][cell] = cs
        for prefix, n in [('n6', 6), ('n8', 8)]:
            for field, lab in [(0, 'accuracy'), (2, 'mean p_gold')]:
                s = decision(cells, n, prefix, field)
                if s is None:
                    continue
                summary[model][f'decision_{prefix}_{lab}'] = {k: v for k, v in s.items()}
                ci = s.get('ci', {})
                out_md.append(f'\n**Decision rule, {prefix}, on {lab}** → class **{s["class"]}**  ')
                out_md.append(f'acc_L(t0) = {s["accL"]:.3f} (CI {ci.get("accL", (0, 0))[0]:.3f}–{ci.get("accL", (0, 0))[1]:.3f}); '
                              f'g: ' + ', '.join(f'{c.split("_", 1)[1]}={v:.2f}' for c, v in s['g'].items()) + '  ')
                out_md.append(f'comp = {s["comp"]:.3f} (CI {ci["comp"][0]:.3f}–{ci["comp"][1]:.3f}); '
                              f'held = {s["held"]:.3f} (CI {ci["held"][0]:.3f}–{ci["held"][1]:.3f}); '
                              f'w = {s["w"]:.2f}; shape g(t4)−g(t1) = {s["shape"]:.3f} (CI {ci["shape"][0]:.3f}–{ci["shape"][1]:.3f})  ')
                r, rc = s['raw'], s['raw_ci']
                out_md.append('raw G contrasts: ' + '; '.join(f'{k} = {r[k]:+.3f} [{rc[k][0]:+.3f},{rc[k][1]:+.3f}]' for k in r))
    txt = '\n'.join(out_md)
    print(txt)
    open(os.path.join(HERE, 'RESULTS_tables.md'), 'w', encoding='utf-8').write('# Experiment 1: result tables (auto-generated by analyze.py)\n' + txt + '\n')
    json.dump(summary, open(os.path.join(HERE, 'results_summary.json'), 'w'), default=float, indent=1)


if __name__ == '__main__':
    main()
