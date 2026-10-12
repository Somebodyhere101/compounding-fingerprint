"""Power analysis / confusion matrix of the pre-registered decision rule (n=6, design k=10).

Predicted per-trial mean accuracies p[learner][measure] are read from ../out/sim_n6.json
(env-averaged over Z6/S3, held-out = S3 after Z6-only history).  A synthetic 'model' of type X answers
each queried cell correctly with probability p_trial, where p_trial ~ Beta with mean (1-lapse)*p + lapse/n
and intra-trial correlation rho; per trial we observe the mean over m cells of the class.

Decision rule (applied to trial means; thresholds fixed in advance):
  acc_L0   = Latin-forced cells, base t=0
  acc_G0   = G-cells (group-forced, not loop-forced), base t=0
  comp     = acc_G(base,t=4)    - max(acc_G(base,t=0), acc_G(scramble,t=4))
  held     = acc_G(heldout,t=4) - max(acc_G(base,t=0), acc_G(scramble,t=4))
  shape    = acc_G(base,t=4)    - acc_G(base,t=1)
  1. acc_L0 < 1/n + 0.25            -> R
  then rescale every accuracy x to g(x) = (x - 1/n)/(acc_L0 - 1/n)  (lapse correction; Latin-forced
  cells are the model's own ceiling; L-cells pooled over all trials of the base t=0 cell)
  2. g(acc_G0) >= 0.85              -> B   (prior already in place at t=0; compounding unobservable)
  3. comp  <  0.15                  -> N
  4. held  >= 0.6*comp              -> A   (else N_iso);  shape is reported, not used.
usage: python power.py
"""
import json
import numpy as np
from report_tables import load, env_curve

n = 6
KD = 10
rng = np.random.default_rng(11)
d = load(6)


def p_of(L, cond, t, cc, tg=None):
    ks, c = env_curve(d, cond, L, t, cc, tg)
    return c[ks.index(KD)]


LEARN = {'R': 'R', 'N': 'N', 'Niso': 'Niso', 'A': 'A', 'B': 'B', 'Bh': 'Bh'}
pred = {}
for X, L in LEARN.items():
    pred[X] = {
        'L0': p_of(L, 'base', 0, 'L'),
        'G0': p_of(L, 'base', 0, 'G'),
        'G1': p_of(L, 'base', 1, 'G'),
        'G4': p_of(L, 'base', 4, 'G'),
        # scramble control: history of identity-free Latin squares carries no group/loop axioms and no
        # matching exemplars -> every idealised non-Bayesian learner behaves as at t=0; the hierarchical
        # Bayesian Bh instead concludes 'Latin environment' (only env with nonzero likelihood) and drops
        # to Latin-level accuracy (we use the Latin possibilistic value as its proxy at n=6)
        'Gs4': p_of('N', 'base', 0, 'G') if L == 'Bh' else p_of(L, 'base', 0, 'G'),
        'Gh4': p_of(L, 'heldout', 4, 'G', ['S3']),
    }
# cells per trial (env-averaged composition at k=10)
mL = 1.30; mG = 11.77


def trial_means(p, m, N, lapse, rho):
    pe = (1 - lapse) * p + lapse / n
    pe = min(max(pe, 1e-4), 1 - 1e-4)
    if rho > 0:
        a = pe * (1 - rho) / rho; b = (1 - pe) * (1 - rho) / rho
        pt = rng.beta(a, b, size=N)
    else:
        pt = np.full(N, pe)
    mm = max(int(round(m)), 1)
    return rng.binomial(mm, pt) / mm


def classify(obs):
    # rule 1 on raw accuracy; rules 2-4 on lapse-corrected accuracy g(x) = (x - 1/n) / (acc_L0 - 1/n),
    # i.e. accuracy rescaled so that chance = 0 and the model's own Latin-forced ceiling = 1
    if obs['L0'] < 1 / n + 0.25: return 'R'
    c = obs['L0'] - 1 / n
    g = lambda x: (x - 1 / n) / c
    if g(obs['G0']) >= 0.85: return 'B'
    base = max(g(obs['G0']), g(obs['Gs4']))   # conservative baseline: t=0 or matched-length scramble
    comp = g(obs['G4']) - base
    if comp < 0.15: return 'N'
    held = g(obs['Gh4']) - base
    return 'A' if held >= 0.6 * comp else 'Niso'


def run(N, lapse, rho, reps=400):
    truth = ['R', 'N', 'Niso', 'A', 'B', 'Bh']
    conf = {}
    for X in truth:
        out = {}
        for _ in range(reps):
            obs = {}
            for key in ('L0', 'G0', 'G1', 'G4', 'Gs4', 'Gh4'):
                m = mL if key == 'L0' else mG
                # L-cells are rare (1.3/trial): pool them over all N trials
                obs[key] = trial_means(pred[X][key], m, N, lapse, rho).mean()
            c = classify(obs)
            out[c] = out.get(c, 0) + 1
        conf[X] = {k: v / reps for k, v in sorted(out.items())}
    return conf


if __name__ == '__main__':
    print('predicted means (n=6, k=10):')
    for X, v in pred.items():
        print('  ', X, {k: round(x, 3) for k, x in v.items()})
    for lapse in (0.0, 0.2, 0.4):
        for rho in (0.3, 0.6):
            for N in (25, 50, 100, 200):
                conf = run(N, lapse, rho)
                acc = np.mean([conf[X].get('B' if X in ('B', 'Bh') else X, 0) for X in conf])
                print(f'lapse={lapse} rho={rho} N={N:4d} trials/cell: mean correct-classification={acc:.3f}  ' +
                      '  '.join(f'{X}->{conf[X]}' for X in conf), flush=True)
