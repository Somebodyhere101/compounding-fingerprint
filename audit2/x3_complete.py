import json, os, glob, collections
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
stim = {os.path.basename(f): {json.loads(l)['trial']: json.loads(l) for l in open(f, encoding='utf-8')} for f in glob.glob(os.path.join(R,'exp1','stim','*.jsonl'))}
nq_exp = sum(len(t['queries']) for s in stim.values() for t in s.values())
for name in ['A1','A2','A4','A6','G4','I4','P4','_A6_step13450','_G4_step9934']:
    rows = [json.loads(l) for l in open(os.path.join(R,'exp3','res',name+'.jsonl'), encoding='utf-8')]
    keys = [(r['file'], r['trial']) for r in rows]
    cells = collections.Counter(r['file'] for r in rows)
    models = set(r['model'] for r in rows)
    nq = sum(len(r['queries']) for r in rows)
    bad = 0
    for r in rows:
        s = stim[r['file']][r['trial']]
        if [q['cell'] for q in r['queries']] != [q['cell'] for q in s['queries']] or [q['gold'] for q in r['queries']] != [q['gold'] for q in s['queries']]: bad += 1
    print(name, 'rows', len(rows), 'unique', len(set(keys)), 'cells', len(cells), 'min/max per cell', min(cells.values()), max(cells.values()), 'models', models, 'queries', nq, '/', nq_exp, 'query-mismatch trials', bad)
