import csv, sys, os, math, collections, json
R = sys.argv[1]
def lr_at(step, lr, warm=500, total=14000, mn=0.1):
    if step < warm: return lr * (step + 1) / warm
    p = min(1.0, (step - warm) / max(1, total - warm)); return lr * (mn + (1 - mn) * 0.5 * (1 + math.cos(math.pi * p)))
for run in ['A1', 'A2', 'A2_stalled', 'A4', 'A6', 'G4', 'I4', 'P4']:
    d = os.path.join(R, 'exp3/runs', run)
    cfg = json.load(open(os.path.join(d, 'config.json')))
    lr = cfg['args']['lr']
    log = list(csv.DictReader(open(os.path.join(d, 'log.csv'))))
    steps = [int(r['step']) for r in log]
    dups = [s for s, c in collections.Counter(steps).items() if c > 1]
    mono = all(b > a for a, b in zip(steps, steps[1:]))
    lrbad = [(int(r['step']), float(r['lr']), lr_at(int(r['step']), lr)) for r in log if abs(float(r['lr']) - lr_at(int(r['step']), lr)) > 0.02 * lr_at(int(r['step']), lr) + 2e-6]
    ev = list(csv.DictReader(open(os.path.join(d, 'eval.csv'))))
    evk = collections.Counter((r['step'], r['cond']) for r in ev)
    evd = [k for k, c in evk.items() if c > 1]
    by = collections.defaultdict(dict)
    for r in ev: by[r['cond']][int(r['step'])] = (float(r['acc']), float(r['q1']))
    print(f'== {run}: last step {steps[-1]}, rows {len(steps)}, monotone {mono}, dup steps {dups[:5]}, lr mismatches {lrbad[:3]}, final loss {log[-1]["loss"]}, min loss {min(float(r["loss"]) for r in log):.4f}')
    print('   eval steps', sorted({int(r['step']) for r in ev}), 'dup eval keys', evd[:3])
    for c in ['in6_t0', 'in6_t4', 'ho6_t4', 'scr6_t4', 'isocp6_t1', 'in8_t4']:
        if c in by:
            s = sorted(by[c]); a = [by[c][x][0] for x in s]; q = [by[c][x][1] for x in s]
            print(f'   {c:10s} acc ' + ' '.join(f'{x:.2f}' for x in a) + f' | max {max(a):.3f} last {a[-1]:.3f} | q1 range {min(q):.2f}-{max(q):.2f}')
