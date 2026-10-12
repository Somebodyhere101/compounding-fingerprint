"""Experiment 6, step 1: check that the re-implemented model + loader reproduce the published behaviour on the model's
OWN training distribution (k=200 facts, final fact held out with its commutative partner). The paper reports convergence
at >= 99% eval accuracy. Also reports accuracy at shorter k (the published model is evaluated with 20..200 facts).
usage: python validate.py [--n 512] -> validate.json"""
import argparse, json, os, random
import numpy as np
import torch
import ica

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=512); ap.add_argument('--ckpt', default='algebra_gpt_best.pt')
a = ap.parse_args()
torch.manual_seed(0)
model = ica.load(a.ckpt)
out = {'ckpt': a.ckpt}
for k in (200, 100, 50, 20):
    prng = random.Random(1000 + k)
    corr, by = [], {}
    for s in range(0, a.n, 64):
        runs = [ica.sample_run(prng, k_shots=k) for _ in range(min(64, a.n - s))]
        x = torch.tensor([r[0] for r in runs], device='cuda')
        with torch.no_grad():
            lg = model(x[:, :-1])[:, -1, :ica.NSYM].float()
        pred = lg.argmax(-1).cpu().numpy(); gold = x[:, -1].cpu().numpy()
        for (toks, groups, held), p, g in zip(runs, pred, gold):
            c = int(p == g); corr.append(c)
            key = '+'.join(f'{t}{m}' for t, m in groups[held[0][0]:held[0][0] + 1])
            by.setdefault(key, []).append(c)
    out[f'k{k}'] = {'acc': float(np.mean(corr)), 'n': len(corr), 'by_group_of_final_fact': {g: [round(float(np.mean(v)), 3), len(v)] for g, v in sorted(by.items())}}
    print(k, out[f'k{k}']['acc'], flush=True)
json.dump(out, open(os.path.join(ica.HERE, f'validate_{os.path.splitext(a.ckpt)[0]}.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
