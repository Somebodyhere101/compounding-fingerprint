"""Paper figures for the compounding-fingerprint study. Re-runnable: uses whatever result files exist.
usage: python figures.py   (CPU env with numpy + matplotlib) -> paper/fig/*.png + *.pdf and paper/fig/figure_data.json
Palette: dataviz reference categorical palette (light mode), fixed slot order, validated set; text in ink tokens."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'exp1'))
import analyze as A1  # noqa: E402

OUT = os.path.join(HERE, 'fig'); os.makedirs(OUT, exist_ok=True)
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
S = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2,
                     'legend.frameon': False, 'savefig.dpi': 200, 'savefig.bbox': 'tight'})
DATA = {}


def save(fig, name):
    for ext in ('png', 'pdf'):
        # no CreationDate in the PDFs, so that rebuilding them is byte-reproducible
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), metadata={'CreationDate': None} if ext == 'pdf' else None)
    plt.close(fig)


def gacc(cells, cell, cls='G'):
    if cell not in cells:
        return np.nan, 0
    return A1.pooled(cells[cell]['trials'], cls)


def boot_contrast(cells, a, b, cls='G', nb=2000, seed=0):
    rng = np.random.default_rng(seed)
    if a not in cells or b not in cells:
        return np.nan, (np.nan, np.nan)
    ta, tb = cells[a]['trials'], cells[b]['trials']
    def pool(tr, idx):
        c = sum(tr[i][cls][0] for i in idx if cls in tr[i]); n = sum(tr[i][cls][1] for i in idx if cls in tr[i])
        return c / n if n else np.nan
    est = pool(ta, range(len(ta))) - pool(tb, range(len(tb)))
    d = [pool(ta, rng.integers(0, len(ta), len(ta))) - pool(tb, rng.integers(0, len(tb), len(tb))) for _ in range(nb)]
    return est, (float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5)))


# ---------------------------------------------------------------- learners and models
IDEAL = [('N', 'Notebook (N)'), ('N_iso', 'Exemplar matching (N$_{iso}$)'), ('A', 'Abstraction (A)'), ('B', 'Bayes, true prior (B)')]
LLMS = [  # (label, path, params in B)
    ('Qwen3-0.6B', 'exp1/res/qwen3-0.6b.jsonl', 0.6), ('Qwen3-0.6B-Base', 'exp2/res/q06b_main.jsonl', 0.6),
    ('Qwen3-1.7B', 'exp1/res/qwen3-1.7b.jsonl', 1.7), ('Qwen3-1.7B-Base', 'exp2/res/q17b_main.jsonl', 1.7),
    ('SmolLM2-1.7B', 'exp1/res/smollm2-1.7b.jsonl', 1.7), ('OLMo-2-1B', 'exp2/res/olmo1b_main.jsonl', 1.5),
    ('Qwen3-4B-Base (nf4)', 'exp2/res/q4b_main.jsonl', 4.0), ('OLMo-2-7B (nf4)', 'exp2/res/olmo7b_main.jsonl', 7.3),
    ('Qwen3-8B-Base (nf4)', 'exp2/res/q8b_main.jsonl', 8.2)]
EXP3 = [('A1', '1 layer, mixed env'), ('A2', '2 layers, mixed env'), ('A4', '4 layers, mixed env'), ('A6', '6 layers, mixed env'),
        ('G4', '4 layers, groups only'), ('P4', '4 layers, fixed pool'), ('I4', '4 layers, isocopy + Latin')]


def complete(path, need=1800):
    p = os.path.join(ROOT, path)
    return os.path.exists(p) and sum(1 for _ in open(p, encoding='utf-8')) >= need


def fig_ideal():
    """Fig 1: idealised learners on the identical trials: G-accuracy vs t, plus held-out."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    DATA['fig1'] = {}
    for ax, n, ch in [(axes[0], 6, 1 / 6), (axes[1], 8, 1 / 8)]:
        ts = [0, 1, 2, 4] if n == 6 else [0, 1, 4]
        for j, ((key, lab), col) in enumerate(zip(IDEAL, S)):
            p = os.path.join(ROOT, 'exp1', 'ideal', 'res', f'ideal_{key}.jsonl')
            if not os.path.exists(p):
                continue
            cells = A1.load(p)
            ys = [gacc(cells, f'n{n}_base_t{t}')[0] for t in ts]
            ho = gacc(cells, f'n{n}_heldout_t4')[0]
            ls, z = (('--', 2) if key == 'B' else ('-', 3))
            ax.plot(ts, ys, ls, marker='o', color=col, ms=5, label=lab, zorder=z)
            ax.plot([4.25 + 0.12 * j], [ho], marker='D', color=col, ms=5, mec=SURF, mew=1.2, zorder=4, ls='none')
            DATA['fig1'][f'{key}_n{n}'] = {'t': ts, 'G': ys, 'heldout_t4': ho}
        ax.axhline(ch, color=MUTED, lw=1, ls='--')
        ax.text(0, ch + 0.015, 'chance', color=MUTED, fontsize=7)
        ax.set_xticks(ts + [4.43]); ax.set_xticklabels([str(t) for t in ts] + ['held\nout'])
        ax.set_title(f'n = {n}', color=INK, fontsize=9, loc='left')
        ax.set_xlabel('earlier tables in context (t)')
    axes[0].set_ylabel('accuracy on associativity-forced cells')
    axes[0].set_ylim(0, 1.05)
    axes[1].legend(loc='center right', fontsize=7)
    save(fig, 'fig1_ideal_learners')


def fig_llm_contrasts():
    """Fig 2: compounding contrasts with 95% CIs for every pretrained model vs idealised references (n=6)."""
    rows = []
    for lab, path, _ in LLMS:
        if not complete(path):
            continue
        cells = A1.load(os.path.join(ROOT, path))
        c1 = boot_contrast(cells, 'n6_base_t4', 'n6_scramble_t4')
        c2 = boot_contrast(cells, 'n6_heldout_t4', 'n6_scramble_t4')
        dec = A1.decision(cells, 6, 'n6', 0, boot=False)
        rows.append((lab, c1, c2, dec['class'] if dec else '?'))
    refs = []
    for key, lab in [('A', 'Abstraction (A)'), ('N_iso', 'Exemplar (N$_{iso}$)'), ('N', 'Notebook (N)')]:
        p = os.path.join(ROOT, 'exp1', 'ideal', 'res', f'ideal_{key}.jsonl')
        if os.path.exists(p):
            cells = A1.load(p)
            refs.append((lab, boot_contrast(cells, 'n6_base_t4', 'n6_scramble_t4'), boot_contrast(cells, 'n6_heldout_t4', 'n6_scramble_t4')))
    if not rows:
        return
    allr = [(r[0] + f'  [{r[3]}]', r[1], r[2], False) for r in rows] + [(r[0] + '  (ideal)', r[1], r[2], True) for r in refs]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.32 * len(allr) + 1.2), sharey=True)
    y = np.arange(len(allr))[::-1]
    for ax, idx, title, col in [(axes[0], 1, 'seen types: t=4 minus scramble', S[0]), (axes[1], 2, 'held-out types: t=4 minus scramble', S[1])]:
        for yi, r in zip(y, allr):
            est, (lo, hi) = r[idx]
            c = MUTED if r[3] else col
            ax.plot([lo, hi], [yi, yi], color=c, lw=2, solid_capstyle='round')
            ax.plot([est], [yi], 'o', color=c, ms=6, mec=SURF, mew=1.5)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(title, fontsize=9, loc='left', color=INK)
        ax.set_xlabel('Δ accuracy on associativity-forced cells (n = 6)')
    axes[0].set_yticks(y); axes[0].set_yticklabels([r[0] for r in allr], fontsize=7.5)
    DATA['fig2'] = [{'label': r[0], 'seen': [r[1][0], *r[1][1]], 'heldout': [r[2][0], *r[2][1]]} for r in allr]
    save(fig, 'fig2_llm_contrasts')


def fig_calibration():
    """Fig 3: Latin-forced accuracy vs cue size k (raw format, pooled over t), per model."""
    order = [('q06b', 'Qwen3-0.6B-Base'), ('q17', 'Qwen3-1.7B'), ('smol17', 'SmolLM2-1.7B'), ('q17b', 'Qwen3-1.7B-Base'),
             ('olmo1b', 'OLMo-2-1B'), ('q4b', 'Qwen3-4B-Base'), ('olmo7b', 'OLMo-2-7B'), ('q8b', 'Qwen3-8B-Base')]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    DATA['fig3'] = {}
    for (key, lab), col in zip(order, S):
        p = os.path.join(ROOT, 'exp2', 'res', f'{key}_cal.jsonl')
        if not os.path.exists(p):
            continue
        agg = {}
        for l in open(p, encoding='utf-8'):
            r = json.loads(l)
            if not r['file'].startswith('ks_'):
                continue
            a = agg.setdefault(r['k'], [0, 0])
            for q in r['queries']:
                if q['class'] == 'L':
                    a[0] += q['correct']; a[1] += 1
        ks = sorted(k for k in agg if agg[k][1] >= 30)
        if not ks:
            continue
        ys = [agg[k][0] / agg[k][1] for k in ks]
        ax.plot(ks, ys, '-o', color=col, ms=4, label=lab)
        DATA['fig3'][lab] = dict(zip(map(str, ks), ys))
    # reference: the from-scratch transformer A4 on the same sweep trials (exp3/res_cal, 2 symbol draws)
    pa = os.path.join(ROOT, 'exp3', 'res_cal', 'A4_cal.jsonl')
    if os.path.exists(pa):
        agg = {}
        for l in open(pa, encoding='utf-8'):
            r = json.loads(l)
            if not r['file'].startswith('ks_'):
                continue
            a = agg.setdefault(r['k'], [0, 0])
            for q in r['queries']:
                if q['class'] == 'L':
                    a[0] += q['correct']; a[1] += 1
        ks = sorted(k for k in agg if agg[k][1] >= 30)
        ys = [agg[k][0] / agg[k][1] for k in ks]
        ax.plot(ks, ys, '--', marker='s', color=INK2, lw=1.4, ms=4, label='A4 (trained, 3.2M)')
        DATA['fig3']['A4 (trained)'] = dict(zip(map(str, ks), ys))
    ax.axhline(1 / 6, color=MUTED, lw=1, ls='--'); ax.text(28.4, 1 / 6 - 0.01, 'chance', color=MUTED, fontsize=7, va='top')
    ax.axhline(0.5, color=INK2, lw=0.8, ls=':'); ax.text(28.4, 0.51, 'k$_L$\nthreshold', color=INK2, fontsize=7)
    ax.set_xticks([10, 16, 22, 28])
    ax.set_xlabel('revealed cells in the target table (k of 36)'); ax.set_ylabel('accuracy on Latin-forced cells')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=6.5, loc='upper left', ncol=2)
    save(fig, 'fig3_calibration')


def fig_exp3():
    """Fig 4: trained transformers: G-accuracy vs t (n=6) and held-out, beside ideal A and N_iso."""
    have = [(k, lab) for k, lab in EXP3 if os.path.exists(os.path.join(ROOT, 'exp3', 'res', f'{k}.jsonl'))]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ts = [0, 1, 2, 4]
    DATA['fig4'] = {}
    for (k, lab), col in zip(have, S):
        cells = A1.load(os.path.join(ROOT, 'exp3', 'res', f'{k}.jsonl'))
        ys = [gacc(cells, f'n6_base_t{t}')[0] for t in ts]
        ho = gacc(cells, 'n6_heldout_t4')[0]
        dec = A1.decision(cells, 6, 'n6', 0, boot=False)
        ax.plot(ts, ys, '-o', color=col, ms=4, label=f'{k}: {lab} [{dec["class"] if dec else "?"}]')
        ax.plot([4.35], [ho], marker='D', color=col, ms=5, mec=SURF, mew=1.5, ls='none')
        DATA['fig4'][k] = {'G': ys, 'heldout_t4': ho, 'class': dec['class'] if dec else None}
    for key, ls, (tx, ty) in [('A', ':', (1.08, 1.015)), ('N_iso', '--', (2.08, 0.80)), ('C', '-.', (2.6, 0.30))]:
        p = os.path.join(ROOT, 'exp1', 'ideal', 'res', f'ideal_{key}.jsonl')
        if os.path.exists(p):
            cells = A1.load(p)
            ax.plot(ts, [gacc(cells, f'n6_base_t{t}')[0] for t in ts], ls, color=MUTED, lw=1.2)
            ax.text(tx, ty, 'ideal C (count shortcut)' if key == 'C' else f'ideal {key}', color=MUTED, fontsize=6.5)
    ax.axhline(1 / 6, color=MUTED, lw=1, ls='--')
    ax.set_xticks(ts + [4.35]); ax.set_xticklabels([str(t) for t in ts] + ['held\nout'])
    ax.set_ylim(0, 1.05); ax.set_xlabel('earlier tables in context (t)'); ax.set_ylabel('accuracy on associativity-forced cells (n = 6)')
    ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
    save(fig, 'fig4_trained_transformers')


def fig_cr():
    """Fig 5: Experiment 5/5b. CR-axis projection per condition minus leak, in units of the pooled SD of leak/base trial means
    (detector format and layer). Leak and leakRel differ only in retrievability; base and baseRel only in vocabulary."""
    p = os.path.join(ROOT, 'exp5', 'results_ctrl.json')
    if not os.path.exists(p):
        return
    J = json.load(open(p))
    labs = {'q06b': 'Qwen3-0.6B-Base', 'q17b': 'Qwen3-1.7B-Base', 'q17': 'Qwen3-1.7B', 'q4b': 'Qwen3-4B-Base (nf4)'}
    conds = [('leak', 'leak (retrievable; reference = 0)'), ('leakRel', 'leakRel (target words, inconsistent)'),
             ('baseRel', 'baseRel (other table, target words)'), ('base', 'base (other table, other words)')]
    names = [k for k in labs if k in J]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    DATA['fig5'] = {}
    w = 0.2
    for j, (c, cl) in enumerate(conds):
        ys = []
        for k in names:
            key = [x for x in J[k] if x.startswith('det_L')][0]  # first det entry = detector layer
            m = J[k][key]['means']
            ys.append((m[c] - m['leak']) / J[k][key]['sd_pooled'])
        ax.bar(np.arange(len(names)) + (j - 1.5) * w, ys, w, color=S[j], label=cl)
        DATA['fig5'][c] = dict(zip(names, ys))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([labs[k] for k in names], fontsize=7)
    ax.set_ylabel('CR projection minus leak\n(SD of trial means)')
    ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, 'fig5_cr_axis_control')


def fig_ica():
    """Fig 6: Experiment 6. (a) L/O/G accuracy at n=6 base t=0 for the In-Context Algebra model vs a pretrained LM, our trained
    network and idealised learners; (b) cue sweep (n=6, t=0, all queried cells) vs learner C and row/column elimination."""
    p = os.path.join(ROOT, 'exp6', 'res', 'ica_best_main.jsonl')
    if not os.path.exists(p):
        return
    rows = [('In-Context Algebra model', p), ('Qwen3-8B-Base (nf4)', os.path.join(ROOT, 'exp2', 'res', 'q8b_main.jsonl')),
            ('A4 (ours, trained)', os.path.join(ROOT, 'exp3', 'res', 'A4.jsonl')),
            ('ideal N (Latin only)', os.path.join(ROOT, 'exp1', 'ideal', 'res', 'ideal_N.jsonl')),
            ('ideal C (count shortcut)', os.path.join(ROOT, 'exp1', 'ideal', 'res', 'ideal_C.jsonl')),
            ('ideal B (group prior)', os.path.join(ROOT, 'exp1', 'ideal', 'res', 'ideal_B.jsonl'))]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 3.0), gridspec_kw={'width_ratios': [1.25, 1]})
    DATA['fig6'] = {'a': {}, 'b': {}}
    w = 0.26
    for j, cls in enumerate(['L', 'O', 'G']):
        ys = []
        for lab, path in rows:
            cells = A1.load(path); ys.append(gacc(cells, 'n6_base_t0', cls)[0])
        ax.bar(np.arange(len(rows)) + (j - 1) * w, ys, w, color=S[j], label=f'{cls}-forced cells')
        DATA['fig6']['a'][cls] = dict(zip([r[0] for r in rows], ys))
    ax.axhline(1 / 6, color=MUTED, lw=1, ls='--')
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], rotation=30, ha='right', fontsize=6.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel('accuracy (n = 6, base, t = 0)'); ax.legend(fontsize=6.5, loc='upper right')
    ax.set_title('(a) forcing-class profile', fontsize=9, loc='left', color=INK)
    J = json.load(open(os.path.join(ROOT, 'exp6', 'results6.json')))['sweep']
    ks = [6, 10, 16, 22, 28]

    def sweep(path):
        out = []
        for k in ks:
            c = n = 0
            for l in open(path, encoding='utf-8'):
                r = json.loads(l)
                if r['file'] == f'ks_n6_k{k}.jsonl' and r['t'] == 0:
                    c += sum(q['correct'] for q in r['queries']); n += len(r['queries'])
            out.append(c / n if n else np.nan)
        return out
    series = [('In-Context Algebra model', sweep(os.path.join(ROOT, 'exp6', 'res', 'ica_best_cal.jsonl')), S[0], '-o'),
              ('Qwen3-8B-Base (nf4)', sweep(os.path.join(ROOT, 'exp2', 'res', 'q8b_cal.jsonl')), S[1], '-o'),
              ('row/column elimination', [J[f'ks_n6_k{k}']['rowcol'] for k in ks], MUTED, '--'),
              ('learner C (count)', [J[f'ks_n6_k{k}']['C'] for k in ks], MUTED, ':')]
    for lab, ys, col, st in series:
        bx.plot(ks, ys, st, color=col, ms=4, lw=1.6 if 'o' in st else 1.2, label=lab)
        DATA['fig6']['b'][lab] = dict(zip(ks, ys))
    bx.axhline(1 / 6, color=MUTED, lw=1, ls='--')
    bx.set_xticks(ks); bx.set_ylim(0, 1.05); bx.set_xlabel('revealed cells k (of 36)'); bx.set_ylabel('accuracy, all queried cells')
    bx.set_title('(b) cue sweep (t = 0)', fontsize=9, loc='left', color=INK); bx.legend(fontsize=6.5, loc='center left', bbox_to_anchor=(0.0, 0.5))
    save(fig, 'fig6_in_context_algebra')


if __name__ == '__main__':
    fig_ideal(); fig_llm_contrasts(); fig_calibration(); fig_exp3(); fig_cr(); fig_ica()
    json.dump(DATA, open(os.path.join(OUT, 'figure_data.json'), 'w'), indent=1, default=float)
    print('figures written:', sorted(os.listdir(OUT)))
