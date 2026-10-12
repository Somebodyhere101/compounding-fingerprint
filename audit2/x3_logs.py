import csv, math, os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'exp3', 'runs')
def lr_at(step, lr, warm=500, steps=14000, mf=0.1):
    if step < warm: return lr*(step+1)/warm
    p = min(1.0, (step-warm)/max(1, steps-warm))
    return lr*(mf+(1-mf)*0.5*(1+math.cos(math.pi*p)))
LR = {'A1':1e-3,'A2':1e-3,'A2_stalled':1e-3,'A4':6e-4,'A6':6e-4,'G4':6e-4,'I4':6e-4,'P4':6e-4}
for r in ['A1','A2','A2_stalled','A4','A6','G4','I4','P4']:
    rows = list(csv.reader(open(os.path.join(R, r, 'log.csv'))))
    hdr = [i for i,x in enumerate(rows) if x and x[0]=='step']
    data = [x for x in rows if x and x[0].isdigit()]
    steps = [int(x[0]) for x in data]
    dup = len(steps)-len(set(steps)); nonmono = sum(1 for a,b in zip(steps,steps[1:]) if b<=a)
    # lr check: logged lr is lr at step-1 (lr computed before increment)
    bad = [(int(x[0]), x[5], f'{lr_at(int(x[0])-1, LR[r]):.2e}') for x in data if x[5] != f'{lr_at(int(x[0])-1, LR[r]):.2e}']
    # elapsed resets indicate restart
    el = [float(x[8]) for x in data]
    resets = [(steps[i], el[i-1], el[i]) for i in range(1,len(el)) if el[i] < el[i-1]]
    gaps = [(steps[i-1], steps[i]) for i in range(1,len(steps)) if steps[i]-steps[i-1] != 50]
    print(f'{r}: headers at {hdr} nrows {len(data)} first {steps[0]} last {steps[-1]} dup {dup} nonmono {nonmono} lr-mismatch {len(bad)} {bad[:3]} elapsed-resets {resets} step-gaps {gaps}')
    ev = list(csv.reader(open(os.path.join(R, r, 'eval.csv'))))
    es = sorted(set(int(x[0]) for x in ev if x and x[0].isdigit()))
    print('   eval steps', es)
