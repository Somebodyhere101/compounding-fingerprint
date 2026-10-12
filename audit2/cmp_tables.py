import json, re, sys, os
R = sys.argv[1]
mine = json.load(open(os.path.join(R, 'audit2/main_res.json')))
lab = {('Qwen3-1.7B-Base', 'nf4 †'): 'Qwen3-1.7B-Base-nf4'}
nums = lambda s: [float(x) for x in re.findall(r'[+-]?\d+\.\d+', s)]
bad = 0; n = 0
for line in open(os.path.join(R, 'paper/results_tables.md'), encoding='utf-8'):
    c = [x.strip() for x in line.strip().strip('|').split('|')]
    if len(c) != 13 or c[0] not in ('llm', 'trained'): continue
    key = lab.get((c[1], c[2]), c[1]); m = mine[key]
    cls = re.findall(r'[\w-]+', c[3]); 
    exp = {'cls_acc': m['n6_f0']['cls'], 'cls_pg': m['n6_f2']['cls'], 'cls8': m['n8_f0']['cls']}
    got = {'cls_acc': cls[0], 'cls_pg': cls[1], 'cls8': c[4]}
    for k in exp:
        if exp[k] != got[k]: print('CLASS MISMATCH', key, k, exp[k], got[k]); bad += 1
    G = m['n6_f0']['G']
    vals = [(m['n6_f0']['accL'], nums(c[5])[0]), (G['base_t0'], nums(c[6])[0]), (G['base_t4'], nums(c[7])[0]), (G['scramble_t4'], nums(c[8])[0]), (G['heldout_t4'], nums(c[9])[0])]
    vals += list(zip(m['n6_t4_scr'], nums(c[10]))) + list(zip(m['n6_held_scr'], nums(c[11]))) + [(m['leak_G'], nums(c[12])[0])]
    for a, b in vals:
        n += 1
        if abs(a - b) > 0.005: print('NUM MISMATCH', key, a, b); bad += 1
    mx = max(abs(a - b) for a, b in vals)
    print(f'{key:22s} max|diff| {mx:.4f}')
print('checked', n, 'numbers; mismatches', bad)
