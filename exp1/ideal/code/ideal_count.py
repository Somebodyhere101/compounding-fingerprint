"""Idealised learner C (count shortcut), added post hoc on 2026-10-09 after the trained networks were found to use it.
For every queried cell, C answers uniformly among the target words that occur least often as a *result* in the revealed cue
(global count only: no row, column, history or axiom information). p_gold = 1/|tied set| if gold is in it, else 0.
Ties are broken with a per-(file, trial, cell) seed. usage: python ideal_count.py -> ../res/ideal_C.jsonl"""
import glob, json, os, re, zlib
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(HERE, '..', '..', 'stim')
FACT = re.compile(r'^(\w+) \* (\w+) = (\w+)$')
out = open(os.path.join(HERE, '..', 'res', 'ideal_C.jsonl'), 'w', encoding='utf-8')
for f in sorted(glob.glob(os.path.join(STIM, '*.jsonl'))):
    fname = os.path.basename(f)
    for tr in map(json.loads, open(f, encoding='utf-8')):
        target = [FACT.match(x).groups() for x in tr['context'].split('Table ')[-1].split('\n')[1:] if FACT.match(x)]
        assert len(target) == tr['k'], (fname, tr['trial'], len(target), tr['k'])
        cnt = Counter(c for _, _, c in target); words = tr['target_words']
        m = min(cnt.get(w, 0) for w in words); low = [w for w in words if cnt.get(w, 0) == m]
        qs = []
        for q in tr['queries']:
            rng = np.random.default_rng([zlib.crc32(fname.encode()), tr['trial'], q['cell'][0], q['cell'][1]])
            pred = low[rng.integers(len(low))]
            qs.append({'cell': q['cell'], 'class': q['class'], 'gold': q['gold'], 'pred': pred, 'correct': int(pred == q['gold']),
                       'p_gold': round((q['gold'] in low) / len(low), 6), 'cand_mass': 1.0})
        out.write(json.dumps({'model': 'ideal_C', 'file': fname, 'trial': tr['trial'], 'cond': tr['cond'], 'n': tr['n'], 't': tr['t'], 'k': tr['k'],
                              'target_type': tr['target_type'], 'ctx_tokens': 0, 'queries': qs}) + '\n')
out.close(); print('DONE ideal_C')
