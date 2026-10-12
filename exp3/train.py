"""Train a small transformer from scratch on in-context group-table completion (Experiment 3).

usage (defaults = proposed default config):
  python train.py --name A_L4 --types noheld --minutes 60
  python train.py --name tiny --device cpu --layers 1 --d_model 32 --heads 2 --orders 4 5 --t_max 1 \
                  --tokens_per_step 2048 --micro_tokens 2048 --steps 50 --eval_every 25 --eval_seqs 4
Writes runs/<name>/{config.json, log.csv, eval.csv, last.pt, final.pt}. Resumes from last.pt with --resume.
"""
import argparse, csv, json, math, os, queue, threading, time, zlib
import numpy as np
import torch
from data import Cfg as DCfg, Gen, parse_env_mix, seq_len, allowed_types, HELDOUT, groups_of_order
from model import GPT, Cfg as MCfg

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_K = {5: 6, 6: 10, 8: 16}   # Exp-1 cue sizes


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--out_root', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    # data
    ap.add_argument('--orders', type=int, nargs='+', default=[4, 5, 6, 7, 8])
    ap.add_argument('--types', default='noheld', choices=['all', 'cyclic', 'abelian', 'noheld'])
    ap.add_argument('--t_min', type=int, default=0)
    ap.add_argument('--t_max', type=int, default=4)
    ap.add_argument('--k_frac', type=float, nargs=2, default=[0.15, 0.45])
    ap.add_argument('--S', type=int, default=64)
    ap.add_argument('--pool_M', type=int, default=0)
    ap.add_argument('--env_mix', default='group=1')
    ap.add_argument('--loss_on', default='all', choices=['all', 'target'])
    ap.add_argument('--max_len', type=int, default=2400)
    # model
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--pos', default='rope', choices=['rope', 'learned'])
    ap.add_argument('--dropout', type=float, default=0.0)
    # optimisation
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--min_lr_frac', type=float, default=0.1)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--steps', type=int, default=20000, help='length of the cosine schedule')
    ap.add_argument('--max_steps', type=int, default=None, help='stop early after this many steps (benchmarks)')
    ap.add_argument('--minutes', type=float, default=None, help='wall-clock budget; checkpoint and stop when hit')
    ap.add_argument('--tokens_per_step', type=int, default=65536)
    ap.add_argument('--micro_tokens', type=int, default=16384, help='tokens per micro-batch (grad accumulation)')
    # logging / eval
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--eval_seqs', type=int, default=64)
    ap.add_argument('--ckpt_every', type=int, default=2000)
    ap.add_argument('--resume', action='store_true')
    return ap.parse_args(argv)


def make_cfgs(a):
    dcfg = DCfg(orders=tuple(a.orders), types=a.types, t_min=a.t_min, t_max=a.t_max, k_frac=tuple(a.k_frac),
                S=a.S, pool_M=a.pool_M, env_mix=parse_env_mix(a.env_mix), max_len=a.max_len,
                loss_on=a.loss_on, seed=a.seed)
    mcfg = MCfg(vocab=6 + a.S, d_model=a.d_model, n_layers=a.layers, n_heads=a.heads, max_len=a.max_len,
                pos=a.pos, dropout=a.dropout)
    return dcfg, mcfg


# ---------------------------------------------------------------- evaluation
def eval_conditions(dcfg):
    """Fixed evaluation conditions: in-distribution, held-out-type and scramble-history targets."""
    conds = []
    for n in (6, 8):
        if seq_len(n, 4) > dcfg.max_len:
            continue
        tr = allowed_types(n, dcfg.types)
        held = [h for h in groups_of_order(n) if (n, h) in HELDOUT]
        for t in (0, 4):
            conds.append((f'in{n}_t{t}', n, t, ('group', tr), ('group', tr)))
            conds.append((f'ho{n}_t{t}', n, t, ('group', tr), ('group', held)))
        conds.append((f'scr{n}_t4', n, 4, ('latin', None), ('group', tr)))
    conds.append(('isocp6_t1', 6, 1, None, 'isocopy'))   # exemplar task: target = relabelled copy of the history table
    return conds


@torch.no_grad()
def evaluate(model, gen, dcfg, device, nseq=64, micro_tokens=16384, seed=999):
    """Teacher-forced accuracy on the target's facts after k=EVAL_K[n] cues, forced choice over the target's n
    symbols. 'acc' = mean over all remaining facts; 'q1' = the first fact after the cue only."""
    model.eval()
    out = {}
    for name, n, t, hist, tgt in eval_conditions(dcfg):
        rng = np.random.default_rng([seed, n, t, zlib.crc32(name.encode())])
        k = EVAL_K[n]
        if tgt == 'isocopy':
            d = gen.make(n, t, nseq, rng, env='isocopy', k=k)
        else:
            d = gen.make(n, t, nseq, rng, hist=hist, tgt=tgt, k=k)
        L = d['tok'].shape[1]
        mb = max(1, micro_tokens // L)
        pos = d['tgt_pos'][k:]                                   # answer positions after the cue
        corr = []; q1 = []
        for s in range(0, nseq, mb):
            tok = torch.from_numpy(d['tok'][s:s + mb]).to(device)
            with autocast(device):
                lg = model(tok[:, :-1])
            lg = lg[:, pos - 1].float()                          # [b, P, V]
            syms = torch.from_numpy(d['tgt_sym'][s:s + mb]).to(device)  # [b, n]
            cl = torch.gather(lg, 2, syms[:, None, :].expand(-1, len(pos), -1))
            pred = syms.gather(1, cl.argmax(-1))                 # [b, P]
            gold = tok[:, pos]
            c = (pred == gold).float()
            corr.append(c.mean(1)); q1.append(c[:, 0])
        out[name] = (float(torch.cat(corr).mean()), float(torch.cat(q1).mean()))
    model.train()
    return out


def autocast(device):
    if str(device).startswith('cuda'):
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return torch.autocast('cpu', enabled=False)


# ---------------------------------------------------------------- batches
def producer(gen, a, seed, q, stop):
    rng = np.random.default_rng([seed, 31337])
    while not stop.is_set():
        try:
            n, t = gen.pick_nt(rng)
            L = seq_len(n, t)
            B = max(1, a.tokens_per_step // L)
            d = gen.make(n, t, B, rng)
            item = (n, t, torch.from_numpy(d['x']), torch.from_numpy(d['y']))
        except Exception as e:  # surface data errors in the main thread
            item = ('ERR', repr(e), None, None)
        while not stop.is_set():
            try:
                q.put(item, timeout=0.5); break
            except queue.Full:
                pass


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def main(argv=None):
    a = build_args(argv)
    out = os.path.join(a.out_root, a.name)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dcfg, mcfg = make_cfgs(a)
    gen = Gen(dcfg)
    dev = torch.device(a.device)
    model = GPT(mcfg).to(dev)
    decay = [p for n_, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n_, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.wd}, {'params': nodecay, 'weight_decay': 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=dev.type == 'cuda')
    step = 0
    ck = os.path.join(out, 'last.pt')
    if a.resume and os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); step = st['step']
        print(f'resumed from step {step}', flush=True)
    json.dump({'args': vars(a), 'data': dcfg.to_dict(), 'model': mcfg.to_dict(), 'nparams': model.nparams()},
              open(os.path.join(out, 'config.json'), 'w'), indent=1)
    print(f'{a.name}: {model.nparams() / 1e6:.2f}M params, device {dev}, (n,t) choices {gen.nt_choices()}', flush=True)
    logf = open(os.path.join(out, 'log.csv'), 'a', newline='')
    evf = open(os.path.join(out, 'eval.csv'), 'a', newline='')
    lw, ew = csv.writer(logf), csv.writer(evf)
    if os.path.getsize(os.path.join(out, 'log.csv')) == 0:
        lw.writerow(['step', 'n', 't', 'B', 'loss', 'lr', 'tok_per_s', 'step_s', 'elapsed_min', 'peak_mem_gb'])
    if os.path.getsize(os.path.join(out, 'eval.csv')) == 0:
        ew.writerow(['step', 'cond', 'acc', 'q1', 'nseq'])

    def save(path):
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step,
                    'mcfg': mcfg.to_dict(), 'dcfg': dcfg.to_dict(), 'args': vars(a)}, path)

    last_eval = [-1]

    def do_eval():
        if last_eval[0] == step:
            return
        last_eval[0] = step
        r = evaluate(model, gen, dcfg, dev, a.eval_seqs, a.micro_tokens)
        for k_, (acc, q1) in r.items():
            ew.writerow([step, k_, f'{acc:.4f}', f'{q1:.4f}', a.eval_seqs])
        evf.flush()
        print(f'[eval step {step}] ' + ' '.join(f'{k_}={v[0]:.3f}' for k_, v in r.items()), flush=True)

    q = queue.Queue(maxsize=6); stop = threading.Event()
    th = threading.Thread(target=producer, args=(gen, a, a.seed + step, q, stop), daemon=True); th.start()
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time(); end = a.steps if a.max_steps is None else min(a.steps, step + a.max_steps)
    acc_loss = 0.0; acc_lab = 0; acc_tok = 0; nlog = 0; t_log = time.time()
    model.train()
    active = 0.0; t_prev = time.time()  # active compute time: gaps > 120 s (laptop sleep/standby) are not counted
    while step < end:
        now = time.time(); active += min(now - t_prev, 120.0); t_prev = now
        if a.minutes is not None and active / 60 > a.minutes:
            print(f'time budget {a.minutes} min reached at step {step}', flush=True)
            break
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g['lr'] = lr
        n, t, X, Y = q.get()
        if n == 'ERR':
            raise RuntimeError('data producer failed: ' + t)
        L = X.shape[1] + 1
        mb = max(1, a.micro_tokens // L)
        nlab = int((Y != -100).sum())
        t0 = time.time()
        tot = 0.0
        for s in range(0, X.shape[0], mb):
            x = X[s:s + mb].to(dev, non_blocking=True); y = Y[s:s + mb].to(dev, non_blocking=True)
            with autocast(dev):
                ls, c = model(x, y)
            (ls / nlab).backward()
            tot += float(ls.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        acc_loss += tot; acc_lab += nlab; acc_tok += X.numel(); nlog += 1
        if step % a.log_every == 0 or step == end:
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            el = time.time() - t_log
            pm = torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else 0.0
            lw.writerow([step, n, t, X.shape[0], f'{acc_loss / max(1, acc_lab):.4f}', f'{lr:.2e}',
                         f'{acc_tok / el:.0f}', f'{el / max(1, nlog):.3f}', f'{(time.time() - t_start) / 60:.2f}', f'{pm:.2f}'])
            logf.flush()
            print(f'step {step} loss {acc_loss / max(1, acc_lab):.4f} lr {lr:.2e} tok/s {acc_tok / el:.0f} '
                  f'peak {pm:.2f}GiB elapsed {(time.time() - t_start) / 60:.1f}min', flush=True)
            acc_loss = 0.0; acc_lab = 0; acc_tok = 0; nlog = 0; t_log = time.time()
        if step % a.eval_every == 0:
            do_eval()
        if step % a.ckpt_every == 0:
            save(ck)
    stop.set()
    do_eval()
    save(ck); save(os.path.join(out, 'final.pt'))
    print(f'DONE {a.name} step {step} elapsed {(time.time() - t_start) / 60:.1f} min (active {active / 60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
