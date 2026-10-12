import csv, os, collections
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'exp3', 'runs')
for r in ['A1','A2','A2_stalled','A4','A6','G4','I4','P4']:
    rows = [x for x in csv.DictReader(open(os.path.join(R, r, 'eval.csv')))]
    d = collections.defaultdict(dict); dup = collections.Counter()
    for x in rows:
        dup[(x['step'], x['cond'])] += 1
        d[x['cond']][int(x['step'])] = float(x['acc'])
    dd = [k for k, v in dup.items() if v > 1]
    def mx(c): return max(d[c].values()) if c in d else float('nan')
    last = max(d['in6_t0'])
    print(f"{r}: dup(step,cond)={len(dd)} {dd[:2]} | max acc in6_t0 {mx('in6_t0'):.3f} in6_t4 {mx('in6_t4'):.3f} ho6_t4 {mx('ho6_t4'):.3f} scr6 {mx('scr6_t4'):.3f} isocp6 {mx('isocp6_t1'):.3f} | final({last}) in6_t0 {d['in6_t0'][last]:.3f} in6_t4 {d['in6_t4'][last]:.3f} ho6_t4 {d['ho6_t4'][last]:.3f} scr6_t4 {d['scr6_t4'][last]:.3f} isocp {d['isocp6_t1'][last]:.3f}  -> learned(>0.267)? {max(mx('in6_t0'), mx('in6_t4'))>0.2667}")
