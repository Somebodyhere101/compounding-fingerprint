import json, os, glob, re
from collections import defaultdict, Counter
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md = open(os.path.join(R, 'exp2/RESULTS2_tables.md'), encoding='utf-8').read()
secs = {m.group(1): m.start() for m in re.finditer(r'## (\w+): cue sweep', md)}
for p in sorted(glob.glob(os.path.join(R, 'exp2/res/*_cal.jsonl'))):
    m = os.path.basename(p)[:-10]
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for l in open(p, encoding='utf-8'):
        r = json.loads(l)
        v = 'instr' if r['file'].startswith('ki_') else 'sorted' if r['file'].startswith('kso_') else 'raw'
        for q in r['queries']:
            a = agg[(v, r['k'], r['t'])][q['class']]; a[0] += q['correct']; a[1] += 1
    start = secs.get(m); end = min([s for s in secs.values() if s > start] + [md.find('main cells')]) if start is not None else None
    sec = md[start:end] if start is not None else ''
    bad = 0
    for key in sorted(agg):
        row = '| ' + ' | '.join([key[0], str(key[1]), str(key[2])] + ['—' if agg[key][c][1] == 0 else f'{agg[key][c][0]/agg[key][c][1]:.3f} (n={agg[key][c][1]})' for c in ['L', 'O', 'G', 'U']]) + ' |'
        if row not in sec: bad += 1; print('  MISMATCH', m, row)
    # L curve
    cur = {}
    for k in sorted({k[1] for k in agg if k[0] == 'raw'}):
        c = sum(agg[('raw', k, t)]['L'][0] for t in (0, 4)); n = sum(agg[('raw', k, t)]['L'][1] for t in (0, 4))
        cur[k] = (round(c / n, 3) if n else None, n)
    print(m, 'section found' if sec else 'NO SECTION', 'rows', len(agg), 'mismatch', bad, 'Lcurve', cur)
# main-cell leak and per-cell completeness for incomplete files
for f in ['q17b_main', 'q8b_main']:
    cnt = Counter(); lk = [0, 0]
    for l in open(os.path.join(R, 'exp2/res', f + '.jsonl'), encoding='utf-8'):
        r = json.loads(l); cnt[r['file']] += 1
        if r['file'] == 'n6_leak_t1.jsonl':
            for q in r['queries']:
                if q['class'] == 'G': lk[0] += q['correct']; lk[1] += 1
    print(f, dict(cnt), 'leak G', lk, lk[0] / lk[1] if lk[1] else None)
