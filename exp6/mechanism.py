"""Experiment 6, exploratory (added after the primary results): where does the model's G-cell accuracy come from?
Splits G-cell accuracy by target type (abelian vs non-abelian) and by whether the transposed cell (b,a) is revealed in the
cue (so 'commutative copying' - answering a*b with the revealed b*a - is available; correct only if the cells agree).
Also: how often the model's answer equals the revealed transpose. -> mechanism.md"""
import json, os, re, sys, zlib
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FACT = re.compile(r'^(\S+) \* (\S+) = (\S+)$')
ABEL = {'Z5', 'Z6', 'Z8', 'Z4xZ2', 'Z2^3'}
stim = {}
def st(f):
    if f not in stim:
        stim[f] = {r['trial']: r for r in map(json.loads, open(os.path.join(ROOT, 'exp1', 'stim', f), encoding='utf-8'))}
    return stim[f]
agg = defaultdict(lambda: [0, 0, 0, 0.0, 0.0, 0.0])   # correct, n, pred==transpose, ideal-N p_gold, ideal-C p_gold, loop-level p
PT = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))   # order -> (file, trial) -> [sum(correct - p_loop), n]
LOOP = {(r['file'], r['trial']): r['p_loop'] for r in map(json.loads, open(os.path.join(HERE, 'res', 'loop_level.jsonl'), encoding='utf-8'))}
IDEAL = {}
for nm in ('N', 'C'):
    for r in map(json.loads, open(os.path.join(ROOT, 'exp1', 'ideal', 'res', f'ideal_{nm}.jsonl'), encoding='utf-8')):
        for q in r['queries']:
            IDEAL[(nm, r['file'], r['trial'], tuple(q['cell']))] = q['p_gold']
for r in map(json.loads, open(os.path.join(HERE, 'res', 'ica_best_main.jsonl'), encoding='utf-8')):
    if 'leak' in r['file']:
        continue
    tr = st(r['file'])[r['trial']]
    cue = {(a, b): c for a, b, c in (FACT.match(x).groups() for x in tr['context'].split('Table ')[-1].split('\n')[1:] if FACT.match(x))}
    for q, s in zip(r['queries'], tr['queries']):
        if q['class'] != 'G':
            continue
        wa, wb = re.findall(r'\S+', s['prompt_suffix'].replace('*', ' ').replace('=', ' '))[:2]
        tt = r['target_type'][1]
        tp = cue.get((wb, wa))
        key = (r['file'][:2], 'abelian' if tt in ABEL else 'non-abelian', 'transpose revealed' if tp else 'transpose hidden',
               '' if tp is None else ('transpose = gold' if tp == q['gold'] else 'transpose != gold'))
        for k in (key, (r['file'][:2], tt, '', '')):
            agg[k][0] += q['correct']; agg[k][1] += 1; agg[k][2] += int(tp is not None and q['pred'] == tp)
            agg[k][3] += IDEAL[('N', r['file'], r['trial'], tuple(q['cell']))]; agg[k][4] += IDEAL[('C', r['file'], r['trial'], tuple(q['cell']))]
            agg[k][5] += LOOP[(r['file'][:-6], r['trial'])][f"{q['cell'][0]},{q['cell'][1]}"]
        if tp is None:   # per-trial sums for the model-minus-loop-level bootstrap
            d = PT[r['file'][:2]][(r['file'], r['trial'])]
            d[0] += q['correct'] - LOOP[(r['file'][:-6], r['trial'])][f"{q['cell'][0]},{q['cell'][1]}"]; d[1] += 1
lines = ['# G-cell accuracy by target structure and transpose availability (post hoc; ica_best_main, stimulus order; pooled over the base t0/t1 and heldout t1 cells of each order)', '',
         'Baselines are expected accuracies on the same cells: ideal N = uniform over Latin-feasible values; loop level = uniform over values a loop (Latin square with two-sided identity) completion allows (res/loop_level.jsonl, loop_level.py); ideal C = count shortcut.', '',
         '| order | target | transpose in cue | transpose vs gold | G acc | n | pred = revealed transpose | ideal N (Latin only) | loop level (exact SAT) | ideal C (count) |', '|---|---|---|---|---|---|---|---|---|---|']
for k in sorted(agg):
    c, n, t, sn, sc, sl = agg[k]
    lines.append(f'| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {c / n:.3f} | {n} | {t / n:.3f} | {sn / n:.3f} | {sl / n:.3f} | {sc / n:.3f} |')
lines += ['', 'Model minus loop level on transpose-hidden G cells (trial bootstrap, 2,000 resamples, seed fixed per order):', '']
for o in sorted(PT):
    v = np.array(list(PT[o].values())); rng = np.random.default_rng(zlib.crc32(o.encode()))
    est = v[:, 0].sum() / v[:, 1].sum(); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(v), len(v)); bs.append(v[i, 0].sum() / v[i, 1].sum())
    lines.append(f'- {o}: {est:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] ({len(v)} trials, {int(v[:, 1].sum())} cells)')
txt = '\n'.join(lines); print(txt); open(os.path.join(HERE, 'mechanism.md'), 'w', encoding='utf-8').write(txt + '\n')
