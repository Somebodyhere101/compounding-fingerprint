"""CPU tests for Experiment 3 (run with the GPU env's python; uses device cpu only).

usage: python tests.py [name ...]      (no names = all tests)
"""
import glob, importlib.util, json, os, shutil, sys, time, traceback
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import data as D
from data import Cfg, Gen, decode, table_from_facts, group_type, seq_len, allowed_types, HELDOUT, NSPEC
from tables import groups_of_order, is_assoc, is_comm, is_latin, identity_of
from model import GPT, Cfg as MCfg


def check_seq(d, cfg, b, hist_spec=None, tgt_spec=None, env=None):
    """Full structural check of sequence b of a batch."""
    n, t = d['n'], d['t']
    tabs = decode(d['tok'][b])
    assert len(tabs) == t + 1, len(tabs)
    used = []
    for j, facts in enumerate(tabs):
        assert len(facts) == n * n
        labs, T = table_from_facts(facts)
        assert len(labs) == n and is_latin(T)
        used.append(set(labs))
        nm = d['names'][b][j]
        if nm in groups_of_order(n):
            ty = group_type(T)
            assert ty == nm, (ty, nm)
            # symbol table equals base table under the recorded injection
            sy = d['syms'][b, j] + NSPEC
            base = groups_of_order(n)[nm]
            for (x, y, z) in facts:
                ia, ib = list(sy).index(x), list(sy).index(y)
                assert sy[base[ia, ib]] == z
        elif nm == 'latin':
            assert identity_of(T) is None
        elif nm == 'loop':
            assert identity_of(T) is not None and not is_assoc(T)
        else:
            raise AssertionError(nm)
    if cfg.pool_M == 0:
        for i in range(len(used)):
            for j in range(i + 1, len(used)):
                assert not (used[i] & used[j]), 'symbol reused across tables'
    # loss mask / answer alignment
    x, y = d['x'][b], d['y'][b]
    lab = np.where(y != -100)[0]
    assert np.all(x[lab] == D.EQ), 'label not at an "=" position'
    assert np.all(y[lab] == d['tok'][b][lab + 1])
    ans_all = [p for p in range(len(d['tok'][b])) if p >= 4 and d['tok'][b][p - 1] == D.EQ]
    if cfg.loss_on == 'all':
        assert list(lab + 1) == ans_all[1:], 'all-but-first-fact mask wrong'
    else:
        k = d['k'][b]
        tp = list(d['tgt_pos'][k:])
        assert list(lab + 1) == tp
    # target metadata
    tp = d['tgt_pos']
    assert all(d['tok'][b][p - 1] == D.EQ for p in tp)
    assert set(d['tok'][b][tp]) <= set(d['tgt_sym'][b]) and len(set(d['tgt_sym'][b])) == n
    # target positions are inside the last table
    last_sep = max([i for i, v in enumerate(d['tok'][b]) if v == D.SEP] or [0])
    assert tp.min() > last_sep


def test_tables_copy():
    spec = importlib.util.spec_from_file_location('c10tables', os.path.join(ROOT, 'runs', 'c10', 'code', 'tables.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for n in range(1, 11):
        A = groups_of_order(n); B = m.groups_of_order(n)
        assert list(A) == list(B)
        for h in A:
            assert np.array_equal(A[h], B[h])
            assert is_latin(A[h]) and is_assoc(A[h]) and identity_of(A[h]) == 0
    return 'exp3/tables.py identical to c10 tables.py for n=1..10; every base table is a group'


def test_data_groups():
    rng = np.random.default_rng(1)
    nseq = 0
    seen = {}
    for types in ['all', 'cyclic', 'abelian', 'noheld']:
        for loss_on in ['all', 'target']:
            cfg = Cfg(orders=(4, 5, 6, 7, 8, 9, 10), types=types, loss_on=loss_on, max_len=3100)
            g = Gen(cfg)
            for n, t in g.nt_choices():
                d = g.make(n, t, 6, rng)
                for b in range(6):
                    check_seq(d, cfg, b); nseq += 1
                    for j in range(t + 1):
                        nm = d['names'][b][j]
                        seen.setdefault(types, set()).add((n, nm))
                        if types == 'cyclic': assert nm == f'Z{n}'
                        if types == 'abelian': assert is_comm(groups_of_order(n)[nm])
                        if types == 'noheld': assert (n, nm) not in HELDOUT
                assert d['tok'].shape[1] == seq_len(n, t) <= cfg.max_len
                lo = int(np.ceil(0.15 * n * n)); hi = int(np.floor(0.45 * n * n))
                assert d['k'].min() >= lo and d['k'].max() <= hi
    allp = {(n, h) for n in (4, 5, 6, 7, 8, 9, 10) for h in groups_of_order(n)}
    assert seen['all'] == allp, allp - seen['all']
    assert seen['noheld'] == allp - HELDOUT
    g = Gen(Cfg(orders=(10,), max_len=3100))
    assert g.tmax[10] == 4
    g = Gen(Cfg(orders=(10,), max_len=2400))
    assert g.tmax[10] == 2, g.tmax
    return f'{nseq} sequences decoded and verified (group type, base table under injection, Latin, fresh symbols, mask)'


def test_data_envs_pool():
    rng = np.random.default_rng(2)
    cfg = Cfg(orders=(4, 5, 6, 8), env_mix={'group': 1, 'latin': 1, 'loop': 1})
    g = Gen(cfg)
    envs = {}
    for n, t in g.nt_choices():
        d = g.make(n, t, 9, rng)
        for b in range(9):
            check_seq(d, cfg, b)
            kinds = {('group' if nm in groups_of_order(n) else nm) for nm in d['names'][b]}
            assert len(kinds) == 1, 'history and target must share the environment'
            kd = kinds.pop(); envs[kd] = envs.get(kd, 0) + 1
    assert set(envs) == {'group', 'latin', 'loop'}, envs
    # isocopy environment: target base table == one of the history base tables, symbols still fresh
    cfg = Cfg(orders=(4, 6, 8), env_mix={'isocopy': 1})
    g = Gen(cfg); ncopy = 0
    for n, t in g.nt_choices():
        d = g.make(n, t, 6, rng)
        for b in range(6):
            check_seq(d, cfg, b)
            assert all(nm == 'latin' for nm in d['names'][b])
            if t >= 1:
                assert any(np.array_equal(d['tabs'][b, t], d['tabs'][b, j]) for j in range(t)); ncopy += 1
    # eval-style override: scramble history, group target
    d = g.make(6, 4, 5, rng, hist=('latin', None), tgt=('group', ['S3']), k=10)
    for b in range(5):
        assert d['names'][b][:4] == ['latin'] * 4 and d['names'][b][4] == 'S3'
        check_seq(d, Cfg(loss_on='all'), b)
    # pool mode
    cfg = Cfg(orders=(4, 6, 8), pool_M=16)
    g = Gen(cfg)
    ov = []
    for n, t in g.nt_choices():
        d = g.make(n, t, 8, rng)
        names_p, tix, syms_p = g.pool[n]
        poolset = {(tuple(s)) for s in syms_p}
        for b in range(8):
            check_seq(d, cfg, b)
            for j in range(t + 1):
                assert tuple(d['syms'][b, j]) in poolset
            u = d['syms'][b].reshape(-1)
            ov.append(len(u) - len(set(u.tolist())))
    distinct = {n: len({tuple(s) for s in g.pool[n][2]}) for n in g.pool}
    return f'envs per sequence {envs}; isocopy targets verified as copies ({ncopy}); pool tables all from pool (pool sizes {distinct}); mean symbol overlap per pool sequence {np.mean(ov):.2f}'


def test_eval_encoding():
    import eval_fp as E
    files = sorted(glob.glob(os.path.join(E.STIM, '*.jsonl')))
    assert len(files) == 18
    ntr = 0; maxL = 0
    for f in files:
        rows = [json.loads(l) for l in open(f, encoding='utf-8')][:5]
        for tr in rows:
            n = tr['n']
            rng = np.random.default_rng(0)
            prefix, qs, w2s = E.encode_trial(tr, 64, rng)
            maxL = max(maxL, len(prefix))
            # fact count and table structure
            tabs = decode(prefix)
            assert len(tabs) == tr['t'] + 1
            for tab in tabs[:-1]:
                labs, T = table_from_facts(tab)
                assert len(labs) == n and is_latin(T)
            htypes = [list(h) for h in tr['history_types']]
            if tr['cond'] == 'leak':
                # Exp-1 quirk: stimuli.py replaces the last history table by the target's own table but keeps the
                # originally drawn type in history_types; the true type of that table is the target's.
                htypes[-1] = list(tr['target_type'])
            for (ht, _), tab in zip(htypes, tabs[:-1]):
                labs, T = table_from_facts(tab)
                if ht == 'grp':
                    assert group_type(T) == _, (f, ht, _)
                if ht == 'latin-noid':
                    assert identity_of(T) is None
                if ht == 'loop':
                    assert identity_of(T) is not None and not is_assoc(T)
            assert len(tabs[-1]) == tr['k']
            # target = cue facts + queries with gold answers must be the complete target table
            full = list(tabs[-1]) + [(q4[0], q4[2], w2s[q['gold']]) for q4, q in zip(qs, tr['queries'])]
            labs, T = table_from_facts(full)
            assert set(labs) == {w2s[w] for w in tr['target_words']}
            if tr['target_type'][0] == 'grp':
                assert group_type(T) == tr['target_type'][1]
            else:
                assert not is_assoc(T) and identity_of(T) is not None
            # distinct words -> distinct symbols; fresh symbols per table except the leak repeat
            assert len(set(w2s.values())) == len(w2s)
            sets = [set(table_from_facts(tab)[0]) for tab in tabs[:-1]] + [set(labs)]
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    if tr['cond'] == 'leak' and i == len(sets) - 2 and j == len(sets) - 1:
                        assert sets[i] == sets[j]
                    else:
                        assert not (sets[i] & sets[j])
            for q4, q in zip(qs, tr['queries']):
                a, b = q['cell']
                assert q4[0] == w2s[tr['target_words'][a]] and q4[2] == w2s[tr['target_words'][b]]
            ntr += 1
    return f'{ntr} Exp-1 trials (5 per file x 18 files) encoded; target cue+gold answers rebuild the declared type; max prefix {maxL} tokens'


def test_packed_equivalence():
    import eval_fp as E
    torch.manual_seed(0)
    out = []
    for pos in ['rope', 'learned']:
        m = GPT(MCfg(vocab=70, d_model=64, n_layers=2, n_heads=4, max_len=300, pos=pos)).eval()
        prefix = [1] + list(np.random.default_rng(0).integers(2, 70, 150))
        qs = [list(np.random.default_rng(i).integers(2, 70, 4)) for i in range(7)]
        with torch.no_grad():
            idx, p, mask, last = E.packed_inputs(prefix, qs, torch.device('cpu'))
            lp = m(idx, pos=p, attn_mask=mask)[0, last]
            sep = torch.stack([m(torch.tensor([prefix + q]))[0, -1] for q in qs])
        err = float((lp - sep).abs().max())
        assert err < 1e-4, err
        out.append(f'{pos} max|diff|={err:.1e}')
    return 'packed independent queries == separate forwards: ' + ', '.join(out)


def test_overfit():
    torch.manual_seed(0)
    cfg = Cfg(orders=(4,), t_max=1)
    g = Gen(cfg)
    d = g.make(4, 1, 8, np.random.default_rng(3))
    x = torch.from_numpy(d['x']); y = torch.from_numpy(d['y'])
    m = GPT(MCfg(vocab=70, d_model=64, n_layers=2, n_heads=4, max_len=256))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    l0 = None
    for it in range(400):
        ls, c = m(x, y)
        loss = ls / c
        if l0 is None: l0 = float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        lg = m(x)
        sel = y != -100
        acc = float((lg.argmax(-1)[sel] == y[sel]).float().mean())
        lf = float(m(x, y)[0] / c)
    assert lf < 0.05 and acc == 1.0, (lf, acc)
    return f'tiny GPT (2L, d64) on one batch of 8 sequences: loss {l0:.3f} -> {lf:.4f}, label accuracy {acc:.3f}'


def test_train_smoke():
    import train
    out = os.path.join(HERE, 'runs', '_cpusmoke')
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    train.main(['--name', '_cpusmoke', '--device', 'cpu', '--layers', '1', '--d_model', '32', '--heads', '2',
                '--orders', '4', '5', '6', '--t_max', '2', '--tokens_per_step', '4096', '--micro_tokens', '2048',
                '--steps', '60', '--warmup', '10', '--log_every', '20', '--eval_every', '30', '--eval_seqs', '4',
                '--ckpt_every', '30', '--env_mix', 'group=0.6,latin=0.2,loop=0.2'])
    for f in ['config.json', 'log.csv', 'eval.csv', 'last.pt', 'final.pt']:
        assert os.path.exists(os.path.join(out, f)), f
    log = open(os.path.join(out, 'log.csv')).read().strip().split('\n')
    ev = open(os.path.join(out, 'eval.csv')).read().strip().split('\n')
    losses = [float(r.split(',')[4]) for r in log[1:]]
    # resume
    train.main(['--name', '_cpusmoke', '--device', 'cpu', '--layers', '1', '--d_model', '32', '--heads', '2',
                '--orders', '4', '5', '6', '--t_max', '2', '--tokens_per_step', '4096', '--micro_tokens', '2048',
                '--steps', '60', '--max_steps', '10', '--warmup', '10', '--log_every', '5', '--eval_every', '1000',
                '--eval_seqs', '4', '--resume', '--minutes', '5', '--env_mix', 'group=0.6,latin=0.2,loop=0.2'])
    st = torch.load(os.path.join(out, 'final.pt'), weights_only=False)
    assert st['step'] == 60, st['step']
    return (f'60 CPU steps in {time.time() - t0:.0f}s; logged losses {losses}; {len(ev) - 1} eval rows '
            f'(11 conditions x evals at steps {sorted({int(r.split(",")[0]) for r in ev[1:]})}); resume restored step 60 and stopped (schedule end)')


def test_eval_fp_end_to_end():
    import eval_fp as E
    sys.path.insert(0, os.path.join(ROOT, 'exp1'))
    import analyze
    ck = os.path.join(HERE, 'runs', '_cpusmoke', 'final.pt')
    assert os.path.exists(ck), 'run test_train_smoke first'
    out = os.path.join(HERE, 'res', '_cputest.jsonl')
    if os.path.exists(out): os.remove(out)
    t0 = time.time()
    E.main(['--ckpt', ck, '--name', '_cputest', '--limit', '3', '--device', 'cpu', '--out', out])
    E.main(['--ckpt', ck, '--name', '_cputest', '--limit', '4', '--device', 'cpu', '--out', out, '--draws', '2'])
    rows = [json.loads(l) for l in open(out, encoding='utf-8')]
    assert len(rows) == 18 * 4, len(rows)
    ref = json.loads(open(os.path.join(ROOT, 'exp1', 'res', 'qwen3-0.6b.jsonl'), encoding='utf-8').readline())
    for r in rows:
        assert set(r) == set(ref), set(r) ^ set(ref)
        for q in r['queries']:
            assert set(q) == set(ref['queries'][0]), set(q)
            assert 0 <= q['p_gold'] <= 1 and 0 < q['cand_mass'] <= 1.00001
    # analyze.py's loader and decision rule accept the file
    cells = analyze.load(out)
    assert len(cells) == 18
    analyze.NBOOT = 50
    s6 = analyze.decision(cells, 6, 'n6', 0, boot=True)
    s8 = analyze.decision(cells, 8, 'n8', 2, boot=True)
    return (f'{len(rows)} trials scored in {time.time() - t0:.0f}s on CPU (resumable; --draws 2 path exercised); '
            f'analyze.load -> {len(cells)} cells; decision n6 class={s6["class"]} accL={s6["accL"]:.2f}, '
            f'n8(p_gold) class={s8["class"]} (untrained tiny model, meaningless values)')


ALL = ['test_tables_copy', 'test_data_groups', 'test_data_envs_pool', 'test_eval_encoding', 'test_packed_equivalence',
       'test_overfit', 'test_train_smoke', 'test_eval_fp_end_to_end']

if __name__ == '__main__':
    names = sys.argv[1:] or ALL
    ok = True
    for nm in names:
        t0 = time.time()
        try:
            msg = globals()[nm]()
            print(f'PASS {nm} ({time.time() - t0:.1f}s): {msg}', flush=True)
        except Exception:
            ok = False
            print(f'FAIL {nm} ({time.time() - t0:.1f}s)', flush=True)
            traceback.print_exc()
    print('ALL PASSED' if ok else 'SOME FAILED')
