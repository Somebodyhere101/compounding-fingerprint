"""Stimulus generator for the compounding-fingerprint experiment (Qwen / Llama pipeline).

Writes JSONL: one line per trial with the prompt text, the queried cells, gold answers, and the exact
forcing class of every queried cell (L=Latin-forced, O=loop-forced, G=group-forced, Ab=abelian-forced,
Cy=cyclic-forced, U=undetermined), computed by exact enumeration / SAT (engine.py).

usage:
  python stimuli.py --n 6 --cond base --t 0 1 2 3 4 --k 8 --trials 200 --out stim_n6_base.jsonl [--vocab words.txt]
  conditions: base | heldout | loops [--loop_target] | scramble | abc | leak | lesion1  (lesion2: see lesion_stage2())
  (lesion2 needs --completed FILE: JSONL from the pipeline holding the model's stage-1 completions)

Token-leak controls implemented here (see report section 5.4):
 C1 fresh vocabulary: no word is used in two tables of the same trial; history words never equal target words.
 C2 random word<->element bijection per table (identity gets a random word); words drawn uniformly per trial.
 C3 facts listed in random order; no row/column headers; tables delimited by neutral headers 'Table A', 'Table B', ...
    (target always gets the next letter, so headers carry no information about table type).
 C4 semantic blacklist (numerals, ordinals, 'one', 'zero', 'unit', 'same', 'identity', 'inverse', ...).
 C5 matched-length control: 'scramble' = random Latin squares WITHOUT an identity element on fresh words,
    same number of facts and tokens per table as the group histories.
 C6 recall positive control 'leak': a history table repeats the target's own labelled table (same words).
 C7 single-token check: filter_vocab(tokenizer) keeps words that are exactly one token with and without a
    leading space; run it in the pipeline before generating.
"""
import argparse, json, itertools, random
import numpy as np
from tables import groups_of_order, relabel, is_comm, is_assoc, all_loops
from engine import universe, sat_solver, counts_for

BLACKLIST = set('''zero one two three four five six seven eight nine ten first second third fourth fifth unit
same identity inverse null none nothing self equal sum product plus minus times half double single pair
twin start end left right top bottom up down yes no true false'''.split())

DEFAULT_WORDS = '''apple river stone cloud tiger maple candle anchor violin garden pepper silver rabbit forest
copper mirror pillow ladder bucket jacket tunnel orange marble pencil window carpet basket button engine
falcon walnut cactus rocket saddle velvet harbor island meadow canyon desert glacier lantern compass helmet
needle shovel hammer bridge castle tower temple cottage cabin barrel kettle bottle spoon fork plate bowl
cup knife table chair sofa lamp clock radio phone camera piano guitar drum flute trumpet ribbon thread
fabric cotton wool leather paper ink chalk brush canvas frame poster ticket letter parcel coin wallet purse
ring necklace crown jewel pearl diamond ruby emerald amber coral shell feather wing beak claw horn tail
fox wolf bear deer moose otter beaver badger eagle hawk owl crow raven swan goose duck parrot pigeon
salmon trout shark whale dolphin seal crab lobster oyster squid snail beetle spider moth butterfly bee
wasp ant cricket lizard turtle frog toad snake cobra python horse donkey camel llama goat sheep cow pig
lemon lime cherry grape melon peach pear plum mango banana carrot potato onion garlic tomato bean corn
wheat rice bread cheese butter honey sugar salt flour cookie cake pie soup salad pasta pizza burger
oven stove fridge sink bath towel soap brush comb razor sock shoe boot glove scarf hat cap coat shirt
dress skirt belt tie vest robe gown blanket sheet quilt mattress curtain shelf drawer closet attic
cellar garage fence gate porch roof chimney wall floor ceiling door hinge lock key chain rope cable
wire pipe hose valve pump gear wheel axle lever pulley spring magnet battery bulb switch plug socket
signal beacon flag banner badge medal trophy prize statue fountain pond lake ocean bay coast cliff hill
valley ridge peak summit cave crater volcano geyser swamp marsh jungle prairie tundra oasis dune storm
thunder rain snow frost hail mist fog breeze wind gale comet planet moon star galaxy nebula orbit
'''.split()


def filter_vocab(words, tokenizer=None):
    """Keep lowercase alphabetic words, not blacklisted, and (if a HF tokenizer is given) exactly one token
    both as ' word' and 'word'."""
    out = []
    seen = set()
    for w in words:
        w = w.strip().lower()
        if not w.isalpha() or w in BLACKLIST or w in seen or not (3 <= len(w) <= 8):
            continue
        if tokenizer is not None:
            a = tokenizer.encode(' ' + w, add_special_tokens=False)
            b = tokenizer.encode(w, add_special_tokens=False)
            if len(a) != 1 or len(b) != 1:
                continue
        seen.add(w)
        out.append(w)
    return out


# ------------------------------------------------------------------ table sources
def random_labelled(T, rng):
    n = len(T)
    s = np.array(rng.permutation(n))
    return relabel(T, s)


def group_table(n, name, rng):
    return random_labelled(groups_of_order(n)[name], rng)


_LOOP_CACHE = {}


def grouplike_loops(n, top=50):
    """Non-associative loops of order n ranked by fraction of associative triples (most group-like first)."""
    if n in _LOOP_CACHE:
        return _LOOP_CACHE[n]
    L = all_loops(n)
    A = L[:, :, :, None]
    res = []
    for X in L:
        lhs = X[X, :]          # (a*b)*c
        rhs = X[:, X]          # a*(b*c)
        frac = float((lhs == rhs).mean())
        if frac < 1.0:
            res.append((frac, is_comm(X), X))
    res.sort(key=lambda r: (-r[0], not r[1]))
    _LOOP_CACHE[n] = res[:top]
    return _LOOP_CACHE[n]


def random_latin_no_identity(n, rng, tries=1000):
    """Random Latin square by randomised backtracking (diverse, not exactly uniform), rejecting squares
    that have a two-sided identity element (so the control carries no loop/group structure)."""
    ar = np.arange(n)
    for _ in range(tries):
        X = -np.ones((n, n), dtype=np.int64)
        def rec(pos):
            if pos == n * n:
                return True
            r, c = divmod(pos, n)
            for v in rng.permutation(n):
                if v in X[r, :c] or v in X[:r, c]:
                    continue
                X[r, c] = v
                if rec(pos + 1):
                    return True
                X[r, c] = -1
            return False
        rec(0)
        has_id = any(np.all(X[e] == ar) and np.all(X[:, e] == ar) for e in range(n))
        if not has_id:
            return X.astype(np.uint8)
    raise RuntimeError('no identity-free square found')


def lesion_stage2(stage1_trial, model_answers, args, rng, vocab):
    """Build the lesion stage-2 trial: context = the model's OWN completed stage-1 target (cue facts plus the
    model's answers for every queried cell, rendered as one complete table), earlier tables removed,
    then a NEW target from the same environment. model_answers: dict 'a,b' -> word (the model's output)."""
    NLc = chr(10)
    n = stage1_trial['n']
    tw = stage1_trial['target_words']
    last_block = stage1_trial['context'].split(NLc + NLc)[-1]
    facts = last_block.split(NLc)[1:]
    for q in stage1_trial['queries']:
        a, b = q['cell']
        facts.append(f"{tw[a]} * {tw[b]} = {model_answers[f'{a},{b}']}")
    rng.shuffle(facts)
    block1 = NLc.join(['Table A:'] + facts)
    H = list(groups_of_order(n).keys())
    tname = rng.choice(H)
    T = group_table(n, tname, rng)
    used = set(tw)
    pool = [w for w in vocab if w not in used]
    w2 = list(rng.choice(pool, size=n, replace=False))
    S = sorted(rng.choice(n * n, size=args.k[0], replace=False).tolist())
    block2 = render_table(T, w2, S, rng, 'Table B:')
    cls = cell_classes(T, S, 'grp', tname)
    queries = [{'cell': [c // n, c % n], 'prompt_suffix': NLc + f'{w2[c // n]} * {w2[c % n]} =',
                'gold': w2[int(T[c // n, c % n])], 'class': cl, 'candidates': w2} for c, cl in cls.items()]
    return {'trial': stage1_trial['trial'], 'cond': 'lesion2', 'n': n, 't': 1, 'k': args.k[0],
            'history_types': [['model-completion', stage1_trial['target_type'][1]]],
            'target_type': ['grp', tname], 'target_words': w2, 'context': block1 + NLc + NLc + block2,
            'queries': queries}



# ------------------------------------------------------------------ forcing classes
def cell_classes(T, S, kind='grp', name=None):
    """Exact forcing class for every unseen cell of target T given cue cells S."""
    n = len(T)
    U = universe(n)
    flat = T.reshape(-1)
    unseen = [c for c in range(n * n) if c not in set(S)]
    S = np.array(sorted(S), dtype=np.int64)

    def forced_enum(keys):
        tot = None
        for k in keys:
            N, c = counts_for(U[k], T, S)
            if N > 0:
                tot = (c > 0) if tot is None else (tot | (c > 0))
        return None if tot is None else (tot.sum(1) == 1)

    def forced_sat(loop):
        if (not loop and 'latin' in U) or (loop and 'loop' in U):
            N, c = counts_for(U['loop' if loop else 'latin'], T, S)
            return (c > 0).sum(1) == 1
        F = sat_solver(n, loop).feasible(T, S)
        return F.sum(1) == 1

    fl = forced_sat(False); fo = forced_sat(True)
    H = list(groups_of_order(n).keys())
    fg = forced_enum(['iso:' + h for h in H]) if kind == 'grp' else None
    if kind == 'grp' and is_comm(groups_of_order(n)[name]):
        fa = forced_enum(['iso:' + h for h in H if is_comm(groups_of_order(n)[h])])
    else:
        fa = None
    fc = forced_enum(['iso:' + name]) if (kind == 'grp' and name == f'Z{n}') else None
    out = {}
    for c in unseen:
        if fl[c]: cl = 'L'
        elif fo[c]: cl = 'O'
        elif fg is not None and fg[c]: cl = 'G'
        elif fa is not None and fa[c]: cl = 'Ab'
        elif fc is not None and fc[c]: cl = 'Cy'
        else: cl = 'U'
        out[c] = cl
    return out


# ------------------------------------------------------------------ rendering
def render_table(T, words, cells, rng, header):
    n = len(T)
    cells = list(cells)
    rng.shuffle(cells)
    lines = [header]
    for c in cells:
        a, b = divmod(int(c), n)
        lines.append(f'{words[a]} * {words[b]} = {words[int(T[a, b])]}')
    return '\n'.join(lines)


def make_trial(args, rng, vocab, trial_id, t, k):
    n = args.n
    H = list(groups_of_order(n).keys())
    seen = {4: ['Z4'], 6: ['Z6'], 8: ['Z8', 'Z4xZ2', 'Z2^3']}.get(n, H)
    held = [h for h in H if h not in seen]
    cond = args.cond
    hist = []  # list of (type, table)
    if cond in ('base', 'leak', 'lesion1'):
        for _ in range(t):
            h = rng.choice(H); hist.append((('grp', h), group_table(n, h, rng)))
    elif cond == 'heldout':
        for _ in range(t):
            h = rng.choice(seen); hist.append((('grp', h), group_table(n, h, rng)))
    elif cond == 'loops':
        pool = grouplike_loops(n)
        for _ in range(t):
            frac, com, X = pool[rng.integers(len(pool))]
            hist.append((('loop', f'assoc={frac:.3f}'), random_labelled(X, rng)))
    elif cond == 'scramble':
        for _ in range(t):
            hist.append((('latin-noid', ''), random_latin_no_identity(n, rng)))
    elif cond == 'abc':
        # phase A: 2 tables from all groups; phase B: 2 abelian tables; target: non-abelian group or loop
        for _ in range(2):
            h = rng.choice(H); hist.append((('grp', h), group_table(n, h, rng)))
        ab = [h for h in H if is_comm(groups_of_order(n)[h])]
        for _ in range(2):
            h = rng.choice(ab); hist.append((('grp', h), group_table(n, h, rng)))
    # target
    if cond == 'heldout':
        tname = rng.choice(held); tkind = 'grp'
    elif cond == 'abc':
        nonab = [h for h in H if not is_comm(groups_of_order(n)[h])]
        tname = rng.choice(nonab) if nonab else rng.choice(H); tkind = 'grp'
    elif cond == 'loops' and args.loop_target:
        tname = None; tkind = 'loop'
    else:
        tname = rng.choice(H); tkind = 'grp'
    if tkind == 'grp':
        T = group_table(n, tname, rng)
    else:
        frac, com, X = grouplike_loops(n)[rng.integers(len(grouplike_loops(n)))]
        T = random_labelled(X, rng)
    # vocabulary: fresh words per table (C1)
    need = n * (len(hist) + 1)
    words = list(rng.choice(vocab, size=need, replace=False))
    tw = words[:n]
    blocks = []
    letters = 'ABCDEFGHIJKLMNOP'
    for i, (typ, X) in enumerate(hist):
        w = words[n * (i + 1): n * (i + 2)]
        if cond == 'leak' and i == len(hist) - 1:
            X, w = T, tw   # C6: repeat the target's own labelled table
        blocks.append(render_table(X, w, range(n * n), rng, f'Table {letters[i]}:'))
    S = sorted(rng.choice(n * n, size=k, replace=False).tolist())
    blocks.append(render_table(T, tw, S, rng, f'Table {letters[len(hist)]}:'))
    context = '\n\n'.join(blocks)
    cls = cell_classes(T, S, 'grp' if tkind == 'grp' else 'loop', tname)
    queries = []
    for c, cl in cls.items():
        a, b = divmod(c, n)
        queries.append({'cell': [a, b], 'prompt_suffix': f'\n{tw[a]} * {tw[b]} =', 'gold': tw[int(T[a, b])],
                        'class': cl, 'candidates': tw})
    return {'trial': trial_id, 'cond': cond, 'n': n, 't': len(hist), 'k': k,
            'history_types': [list(tp) for tp, _ in hist], 'target_type': [tkind, tname],
            'target_words': tw, 'context': context, 'queries': queries,
            'scoring': 'forced choice over target_words: argmax of next-token log-prob of " "+word'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--cond', default='base')
    ap.add_argument('--t', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--k', type=int, nargs='+', default=[8])
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--loop_target', action='store_true')
    ap.add_argument('--vocab', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='stimuli.jsonl')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    vocab = filter_vocab(open(args.vocab).read().split() if args.vocab else DEFAULT_WORDS)
    tid = 0
    with open(args.out, 'w') as f:
        for t in args.t:
            for k in args.k:
                for _ in range(args.trials):
                    tr = make_trial(args, rng, vocab, tid, t, k)
                    f.write(json.dumps(tr) + '\n')
                    tid += 1
    print('wrote', tid, 'trials to', args.out, '| vocab size', len(vocab))


if __name__ == '__main__':
    main()
