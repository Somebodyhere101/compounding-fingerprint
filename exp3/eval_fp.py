"""Score an Experiment-3 checkpoint on the EXACT Experiment-1 stimulus files; write Exp-1 results format.

usage: python eval_fp.py --ckpt runs/A_L4/final.pt --name A_L4 [--files ../exp1/stim/n6_*.jsonl] [--limit N]
       [--device cuda] [--draws 1] [--seed 0]
Output: res/<name>.jsonl, one line per trial with the same fields as exp1/run.py:
  model, file, trial, cond, n, t, k, target_type, ctx_tokens, queries[{cell, class, gold, pred, correct, p_gold, cand_mass}]
Analyse with:  python analyze3.py   (runs exp1/analyze.py's own code on exp3/res/)

Word -> symbol mapping: every distinct word of a trial gets a random distinct symbol (seeded by file+trial+draw).
Exp-1 stimuli use fresh words per table (control C1), so this is a fresh random symbol assignment per table,
consistent within a table; in the 'leak' control the repeated table keeps its words and hence its symbols, as in
Exp-1. 'Table X:' headers become the table separator '|' (none before the first table). Each fact
'w1 * w2 = w3' becomes 'a * b = c ;'. Each query 'wa * wb =' is appended as 'a * b =' and scored independently:
all queries are packed after one copy of the context, each query block attends to the whole context and to
itself only, and re-uses positions L..L+3, so it is exactly equivalent to scoring context+query alone.
Scoring is forced choice by the next-token logits restricted to the n candidate symbols (p_gold = softmax over
candidates; cand_mass = total full-vocabulary probability on the candidates). With --draws D > 1 the candidate
probabilities and cand_mass are averaged over D independent symbol assignments.
"""
import argparse, glob, json, os, re, time, zlib
import numpy as np
import torch
from data import BOS, MUL, EQ, SEMI, SEP, NSPEC
from model import GPT, Cfg as MCfg

HERE = os.path.dirname(os.path.abspath(__file__))
STIM = os.path.join(os.path.dirname(HERE), 'exp1', 'stim')
FACT_RE = re.compile(r'^(\S+) \* (\S+) = (\S+)$')
QUERY_RE = re.compile(r'^\n(\S+) \* (\S+) =$')


def parse_context(ctx):
    """-> list of tables, each a list of (w1, w2, w3)."""
    tables = []
    for line in ctx.split('\n'):
        if not line.strip():
            continue
        if line.startswith('Table '):
            assert line.endswith(':'), line
            tables.append([])
            continue
        m = FACT_RE.match(line)
        assert m and tables, line
        tables[-1].append(m.groups())
    return tables


def trial_words(tr, tables):
    seen = {}
    for tab in tables:
        for f in tab:
            for w in f:
                seen.setdefault(w, None)
    for w in tr['target_words']:
        seen.setdefault(w, None)
    return list(seen)


def encode_trial(tr, S, rng):
    """-> (prefix token list, list of 4-token query lists, word->token map)."""
    tables = parse_context(tr['context'])
    words = trial_words(tr, tables)
    assert len(words) <= S, f'{len(words)} distinct words > {S} symbols'
    syms = rng.permutation(S)[:len(words)] + NSPEC
    w2s = {w: int(s) for w, s in zip(words, syms)}
    tok = [BOS]
    for i, tab in enumerate(tables):
        if i:
            tok.append(SEP)
        for a, b, c in tab:
            tok += [w2s[a], MUL, w2s[b], EQ, w2s[c], SEMI]
    qs = []
    for q in tr['queries']:
        m = QUERY_RE.match(q['prompt_suffix'])
        assert m, q['prompt_suffix']
        wa, wb = m.groups()
        qs.append([w2s[wa], MUL, w2s[wb], EQ])
    return tok, qs, w2s


def packed_inputs(prefix, qs, device):
    """One sequence = prefix + all query blocks; position ids and attention mask so each block sees prefix+itself."""
    L = len(prefix); Q = len(qs); T = L + 4 * Q
    idx = torch.tensor(prefix + [x for q in qs for x in q], device=device)[None]
    pos = torch.cat([torch.arange(L), (L + torch.arange(4)).repeat(Q)]).to(device)
    mask = torch.zeros(T, T, dtype=torch.bool)
    mask[:L, :L] = torch.tril(torch.ones(L, L, dtype=torch.bool))
    mask[L:, :L] = True
    blk = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    for j in range(Q):
        s = L + 4 * j
        mask[s:s + 4, s:s + 4] = blk
    last = torch.tensor([L + 4 * j + 3 for j in range(Q)], device=device)
    return idx, pos, mask.to(device), last


def load_model(ckpt, device):
    st = torch.load(ckpt, map_location=device, weights_only=False)
    model = GPT(MCfg(**st['mcfg'])).to(device).eval()
    model.load_state_dict(st['model'])
    return model, st


@torch.no_grad()
def score_trial(model, tr, S, seed_key, draws, device):
    n = tr['n']
    pcs = None; masses = None; L0 = None
    for d in range(draws):
        rng = np.random.default_rng([*seed_key, d])
        prefix, qs, w2s = encode_trial(tr, S, rng)
        L0 = len(prefix)
        idx, pos, mask, last = packed_inputs(prefix, qs, device)
        ac = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else torch.autocast('cpu', enabled=False)
        with ac:
            lg = model(idx, pos=pos, attn_mask=mask)[0, last].float()       # [Q, V]
        lp = torch.log_softmax(lg, -1)
        cids = torch.tensor([[w2s[w] for w in q['candidates']] for q in tr['queries']], device=device)  # [Q, n]
        cl = lp.gather(1, cids)
        pc = torch.softmax(cl, -1)
        cm = torch.exp(cl).sum(-1)
        pcs = pc if pcs is None else pcs + pc
        masses = cm if masses is None else masses + cm
    pcs = (pcs / draws).cpu().numpy(); masses = (masses / draws).cpu().numpy()
    recs = []
    for i, q in enumerate(tr['queries']):
        j = int(np.argmax(pcs[i])); gi = q['candidates'].index(q['gold'])
        recs.append({'cell': q['cell'], 'class': q['class'], 'gold': q['gold'], 'pred': q['candidates'][j],
                     'correct': int(j == gi), 'p_gold': round(float(pcs[i, gi]), 5),
                     'cand_mass': round(float(masses[i]), 5)})
    return recs, L0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--name', required=True, help='output res/<name>.jsonl (prefix "_" = excluded by analyze)')
    ap.add_argument('--files', nargs='*', default=None)
    ap.add_argument('--limit', type=int, default=None, help='max trials per stimulus file')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--draws', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    model, st = load_model(a.ckpt, dev)
    S = st['dcfg']['S']; maxlen = st['mcfg']['max_len']
    files = a.files or sorted(glob.glob(os.path.join(STIM, '*.jsonl')))
    out = a.out or os.path.join(HERE, 'res', a.name + '.jsonl')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for l in open(out, encoding='utf-8'):
            r = json.loads(l); done.add((r['file'], r['trial']))
    fout = open(out, 'a', encoding='utf-8')
    t0 = time.time(); ntr = 0; nq = 0
    for f in files:
        fname = os.path.basename(f)
        rows = [json.loads(l) for l in open(f, encoding='utf-8')]
        if a.limit:
            rows = rows[:a.limit]
        for tr in rows:
            if (fname, tr['trial']) in done:
                continue
            recs, L = score_trial(model, tr, S, [a.seed, zlib.crc32(fname.encode()), tr['trial']], a.draws, dev)
            assert L + 4 <= maxlen, f'context {L} too long for max_len {maxlen}'
            fout.write(json.dumps({'model': a.name, 'file': fname, 'trial': tr['trial'], 'cond': tr['cond'],
                                   'n': tr['n'], 't': tr['t'], 'k': tr['k'], 'target_type': tr['target_type'],
                                   'ctx_tokens': L, 'queries': recs}) + '\n')
            fout.flush()
            ntr += 1; nq += len(recs)
            if ntr % 100 == 0:
                print(f'{a.name} {fname} trials={ntr} queries={nq} {time.time() - t0:.0f}s', flush=True)
    print(f'DONE {a.name}: trials={ntr} queries={nq} time={time.time() - t0:.0f}s -> {out}', flush=True)


if __name__ == '__main__':
    main()
