"""Experiment 6: score the In-Context Algebra checkpoint on our stimulus files (see PREREGISTRATION.md).
usage: gpy.sh score.py --files ../exp1/stim/n6_base_t0.jsonl ... --out res/ica_best.jsonl [--ckpt algebra_gpt_best.pt]
       [--draws 4] [--interleave] [--t0_only] [--limit N]
Resumable at trial level. Output records follow the Experiment-1 results format (exp1/analyze.load reads them)."""
import argparse, json, os, re, time, zlib
import numpy as np
import torch
import ica

FACT = re.compile(r'^(\S+) \* (\S+) = (\S+)$')
QUERY = re.compile(r'^\n(\S+) \* (\S+) =$')


def parse_tables(ctx):
    tabs = []
    for line in ctx.split('\n'):
        if line.startswith('Table '):
            tabs.append([])
        elif line.strip():
            m = FACT.match(line); assert m and tabs, line
            tabs[-1].append(m.groups())
    return tabs


def identity_of(facts):
    """element e with e*x = x and x*e = x for every x of a COMPLETE table, else None"""
    tab = {(a, b): c for a, b, c in facts}
    els = sorted({a for a, _, _ in facts} | {b for _, b, _ in facts})
    if len(tab) != len(els) ** 2:
        return None
    for e in els:
        if all(tab[(e, x)] == x and tab[(x, e)] == x for x in els):
            return e
    return None


def encode(tr, draw, interleave, fname):
    tabs = parse_tables(tr['context'])
    target_full = list(tabs[-1]) + [tuple(QUERY.match(q['prompt_suffix']).groups()) + (q['gold'],) for q in tr['queries']]
    ids = [identity_of(t) for t in tabs[:-1]] + [identity_of(target_full)]
    words = []
    for t in tabs:
        for f in t:
            for w in f:
                if w not in words:
                    words.append(w)
    for w in tr['target_words']:
        if w not in words:
            words.append(w)
    assert len(words) <= ica.NSYM, (fname, tr['trial'], len(words))
    rng = np.random.default_rng([zlib.crc32(fname.encode()), tr['trial'], draw, 6])
    while True:
        perm = rng.permutation(ica.NSYM)[:len(words)]
        s = {w: int(perm[i]) for i, w in enumerate(words)}
        if not any(e is not None and s[e] == 0 for e in ids):
            break
    facts = [f for t in tabs for f in t]
    if interleave:
        facts = [facts[i] for i in np.random.default_rng([zlib.crc32(fname.encode()), tr['trial'], draw, 66]).permutation(len(facts))]
    prefix = []
    for a, b, c in facts:
        prefix += [ica.SEP, s[a], s[b], ica.EQ, s[c]]
    qs = [[ica.SEP, s[wa], s[wb], ica.EQ] for wa, wb in (QUERY.match(q['prompt_suffix']).groups() for q in tr['queries'])]
    return prefix, qs, s, ids


@torch.no_grad()
def score_trial(model, tr, draws, interleave, fname):
    pcs = masses = None
    for d in range(draws):
        prefix, qs, s, _ = encode(tr, d, interleave, fname)
        x = torch.tensor([prefix + q for q in qs], device='cuda')
        assert x.shape[1] <= model.block
        lp = torch.log_softmax(model(x)[:, -1].float(), -1)
        cids = torch.tensor([[s[w] for w in q['candidates']] for q in tr['queries']], device='cuda')
        cl = lp.gather(1, cids)
        pc, cm = torch.softmax(cl, -1), torch.exp(cl).sum(-1)
        pcs = pc if pcs is None else pcs + pc
        masses = cm if masses is None else masses + cm
    pcs = (pcs / draws).cpu().numpy(); masses = (masses / draws).cpu().numpy()
    recs = []
    for i, q in enumerate(tr['queries']):
        j = int(np.argmax(pcs[i])); gi = q['candidates'].index(q['gold'])
        recs.append({'cell': q['cell'], 'class': q['class'], 'gold': q['gold'], 'pred': q['candidates'][j],
                     'correct': int(j == gi), 'p_gold': round(float(pcs[i, gi]), 5), 'cand_mass': round(float(masses[i]), 5)})
    return recs, len(prefix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default='algebra_gpt_best.pt'); ap.add_argument('--draws', type=int, default=4)
    ap.add_argument('--interleave', action='store_true'); ap.add_argument('--t0_only', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    model = ica.load(a.ckpt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = {(r['file'], r['trial']) for r in map(json.loads, open(a.out, encoding='utf-8'))} if os.path.exists(a.out) else set()
    fout = open(a.out, 'a', encoding='utf-8'); t0 = time.time(); n = 0
    for f in a.files:
        fname = os.path.basename(f)
        rows = [json.loads(l) for l in open(f, encoding='utf-8')]
        if a.t0_only:
            rows = [r for r in rows if r['t'] == 0]
        for tr in rows[:a.limit]:
            if (fname, tr['trial']) in done:
                continue
            recs, L = score_trial(model, tr, a.draws, a.interleave, fname)
            fout.write(json.dumps({'model': f'ica:{a.ckpt}', 'file': fname, 'trial': tr['trial'], 'cond': tr['cond'], 'n': tr['n'],
                                   't': tr['t'], 'k': tr['k'], 'target_type': tr['target_type'], 'ctx_tokens': L,
                                   'format': 'interleaved' if a.interleave else 'stimulus_order', 'draws': a.draws,
                                   'queries': recs}) + '\n')
            fout.flush(); n += 1
        print(f'{fname} done ({n} trials, {time.time() - t0:.0f}s)', flush=True)
    print(f'DONE {a.out} {n} trials {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
