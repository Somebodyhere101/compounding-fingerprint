import json, os, collections
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
fd = json.load(open(os.path.join(R,'paper','fig','figure_data.json')))
def g(path, f, cls='G'):
    c = n = 0
    for l in open(path, encoding='utf-8'):
        r = json.loads(l)
        if r['file'] != f: continue
        for q in r['queries']:
            if q['class'] == cls: c += q['correct']; n += 1
    return c / n
for k in ['A1','A2','A4','A6','G4','I4','P4','_A6_step13450','_G4_step9934']:
    p = os.path.join(R,'exp3','res',k+'.jsonl')
    vals = [g(p, f) for f in ['n6_base_t0.jsonl','n6_base_t4.jsonl','n6_scramble_t4.jsonl','n6_heldout_t4.jsonl','n6_leak_t1.jsonl']]
    accL = g(p, 'n6_base_t0.jsonl', 'L')
    print(k, 'accL %.3f' % accL, 'G t0/t4/scr4/held4/leak', ' '.join('%.3f' % v for v in vals), '| fig4', fd.get('fig4', {}).get(k.strip('_').split('_')[0] if not k.startswith('_') else 'X', {}).get('G'))
