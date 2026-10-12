import json, sys, os, numpy as np
A = os.path.dirname(os.path.abspath(__file__))
for name in ['q06b', 'q17b']:
    recs = json.load(open(os.path.join(A, f'e5_gpu_{name}.json')))
    BL = {'q06b': 26, 'q17b': 26}[name]
    for fmt in ('det', 'raw'):
        for li in sorted(set(r['layer'] for r in recs)):
            g = {}
            for r in recs:
                if r['fmt'] == fmt and r['layer'] == li:
                    g.setdefault((r['file'], r['trial']), []).append(r)
            s_all, u_all, sh_all = [], [], []
            for k, rr in g.items():
                rr = sorted(rr, key=lambda r: r['j'])
                s = np.array([r['stored'] for r in rr]); u = np.array([r['uncached'] for r in rr])
                s_all += list(s - s.mean()); u_all += list(u - u.mean()); sh_all += list(np.roll(u, 1) - u.mean())
            s_all, u_all, sh_all = map(np.array, (s_all, u_all, sh_all))
            print(f'{name} {fmt} L{li}{"*" if li==BL else ""}: within-trial corr(stored,own uncached)={np.corrcoef(s_all,u_all)[0,1]:.4f}  vs shifted-by-1={np.corrcoef(s_all,sh_all)[0,1]:.4f}  within-trial sd={u_all.std():.4g} rms diff={np.sqrt(np.mean((s_all-u_all)**2)):.4g}')
