import json, os, glob, sys
from collections import Counter, defaultdict
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# load stimuli
stim = {}
for d in ['exp1/stim', 'exp2/stim']:
    for p in glob.glob(os.path.join(R, d, '*.jsonl')):
        f = os.path.basename(p)
        for l in open(p, encoding='utf-8'):
            s = json.loads(l)
            stim[(d, f, s['trial'])] = s
files = []
for d in ['exp1/res', 'exp2/res', 'exp1/ideal/res', 'exp3/res']:
    files += sorted(glob.glob(os.path.join(R, d, '*.jsonl')))
for p in files:
    rel = os.path.relpath(p, R).replace(os.sep, '/')
    raw = open(p, 'rb').read()
    endnl = raw.endswith(b'\n')
    bad = 0; keys = Counter(); nq = 0; mism = Counter(); fileset = Counter()
    lines = raw.split(b'\n')
    if lines and lines[-1] == b'': lines = lines[:-1]
    for i, l in enumerate(lines):
        try:
            r = json.loads(l)
        except Exception as e:
            bad += 1; print('  BAD line', rel, i, len(l), str(e)[:80]); continue
        if 'queries' not in r:
            continue
        keys[(r['file'], r['trial'])] += 1
        fileset[r['file']] += 1
        sd = 'exp1/stim' if (('exp1/stim', r['file'], r['trial']) in stim) else 'exp2/stim'
        s = stim.get((sd, r['file'], r['trial']))
        if s is None:
            mism['nostim'] += 1; continue
        sq = {tuple(q['cell']): q for q in s['queries']}
        rq = {tuple(q['cell']): q for q in r['queries']}
        if len(rq) != len(r['queries']): mism['dupcell'] += 1
        if set(sq) != set(rq): mism['cellset'] += 1
        for c, q in rq.items():
            nq += 1
            if c not in sq: continue
            if sq[c]['class'] != q['class']: mism['class'] += 1
            if sq[c]['gold'] != q['gold']: mism['gold'] += 1
            if int(q['pred'] == q['gold']) != q['correct']: mism['correct'] += 1
            if q['pred'] not in sq[c]['candidates']: mism['predcand'] += 1
        for k in ['n', 't', 'k', 'cond']:
            if k in s and r.get(k) != s[k]: mism['meta_' + k] += 1
    dups = sum(v - 1 for v in keys.values() if v > 1)
    print(f'{rel}: lines={len(lines)} endnl={endnl} bad={bad} uniq={len(keys)} dups={dups} queries={nq} mism={dict(mism)} files={len(fileset)} percell={sorted(set(fileset.values()))}')
