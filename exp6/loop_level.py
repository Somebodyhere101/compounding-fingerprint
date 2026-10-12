"""Experiment 6, post hoc (added after the verification pass, 2026-10-11): loop-level baseline for G cells.
For every scored query of the main cells, the expected accuracy of a learner that picks uniformly among the values a
LOOP (Latin square with a two-sided identity) completion of the cue allows at that cell (exact, SAT via the stimulus
generator's own engine). G cells are by definition not loop-forced, so a model above this level uses group structure
beyond identity + cancellation. -> res/loop_level.jsonl (one line per trial: file, trial, {cell: p_loop})"""
import json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'runs', 'c10', 'code'))
from engine import sat_solver  # noqa: E402
FACT = re.compile(r'^(\S+) \* (\S+) = (\S+)$')
CELLS = ['n5_base_t0', 'n6_base_t0', 'n6_base_t1', 'n6_heldout_t1', 'n8_base_t0', 'n8_base_t1', 'n8_heldout_t1']
out_p = os.path.join(HERE, 'res', 'loop_level.jsonl')
done = {(r['file'], r['trial']) for r in map(json.loads, open(out_p, encoding='utf-8'))} if os.path.exists(out_p) else set()
fout = open(out_p, 'a', encoding='utf-8')
for cell in CELLS:
    for tr in map(json.loads, open(os.path.join(ROOT, 'exp1', 'stim', cell + '.jsonl'), encoding='utf-8')):
        if (cell, tr['trial']) in done:
            continue
        n = tr['n']; idx = {w: i for i, w in enumerate(tr['target_words'])}
        cue = [FACT.match(x).groups() for x in tr['context'].split('Table ')[-1].split('\n')[1:] if FACT.match(x)]
        T = np.zeros((n, n), dtype=np.int64)
        for a, b, c in cue:
            T[idx[a], idx[b]] = idx[c]
        for q in tr['queries']:
            T[q['cell'][0], q['cell'][1]] = idx[q['gold']]
        S = sorted(idx[a] * n + idx[b] for a, b, _ in cue)
        F = sat_solver(n, True).feasible(T, np.array(S))
        assert F is not None
        rec = {}
        for q in tr['queries']:
            cflat = q['cell'][0] * n + q['cell'][1]
            assert F[cflat, idx[q['gold']]]
            rec[f"{q['cell'][0]},{q['cell'][1]}"] = round(1.0 / F[cflat].sum(), 6)
        fout.write(json.dumps({'file': cell, 'trial': tr['trial'], 'p_loop': rec}) + '\n'); fout.flush()
    print(cell, 'done', flush=True)
print('DONE loop_level')
