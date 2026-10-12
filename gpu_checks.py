"""Post-audit GPU checks (2026-10-09). Idempotent: every step resumes at trial level. Log: gpu_checks.log
1. Offload-path equivalence (archived this time): Qwen3-1.7B-Base bf16, standard vs --offload_head; Qwen3-4B-Base nf4, standard
   vs nf4 + offload (the 8B configuration). batch 1, first 4 (1.7B) / 2 (4B) trials of every Experiment-1 cell.
2. Qwen3-1.7B-Base primary-contrast cells re-scored with one code version and fixed B=1 (n6 base t0/t4, scramble t4, heldout t4).
3. Experiment 5b vocabulary-overlap control (exp5/CONTROL_vocab.md).
"""
import ctypes, os, subprocess, time

ROOT = os.path.dirname(os.path.abspath(__file__))
GPY = r'C:\Users\%USERNAME%\AppData\Local\Temp\cgpu\Scripts\python.exe'
LOG = os.path.join(ROOT, 'gpu_checks.log')
S1 = '../exp1/stim'


def log(msg):
    line = f'{time.strftime("%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    open(LOG, 'a', encoding='utf-8').write(line + '\n')


def run(label, cmd, cwd):
    for attempt in range(1, 4):
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
        log(f'START {label} (attempt {attempt})')
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        out = (p.stdout or '') + (p.stderr or '')
        for l in [l for l in out.splitlines() if any(k in l for k in ('DONE', 'Error', 'Traceback', 'loaded'))][-4:]:
            log(f'  {l[:240]}')
        if any(k in out for k in ('Application Control', 'DLL load failed')):
            log(f'RETRY {label}: code-integrity block'); time.sleep(120); continue
        log(f'END {label} rc={p.returncode}')
        return p.returncode == 0
    log(f'GAVE UP {label}')


def main():
    e2, e5 = os.path.join(ROOT, 'exp2'), os.path.join(ROOT, 'exp5')
    base = [GPY, '-u', 'run.py', '--stimdir', S1, '--batch', '1', '--mem_frac', '0.6']
    run('offchk 1.7B std', base + ['--model', 'Qwen/Qwen3-1.7B-Base', '--limit', '4', '--out', 'res/_offchk_q17b_std.jsonl'], e2)
    run('offchk 1.7B offload', base + ['--model', 'Qwen/Qwen3-1.7B-Base', '--limit', '4', '--offload_head', '--out', 'res/_offchk_q17b_off.jsonl'], e2)
    run('offchk 4B nf4 std', base + ['--model', 'Qwen/Qwen3-4B-Base', '--quant', 'nf4', '--limit', '2', '--out', 'res/_offchk_q4b_std.jsonl'], e2)
    run('offchk 4B nf4 offload', base + ['--model', 'Qwen/Qwen3-4B-Base', '--quant', 'nf4', '--limit', '2', '--offload_head', '--out', 'res/_offchk_q4b_off.jsonl'], e2)
    files = [f'{S1}/{f}.jsonl' for f in ('n6_base_t0', 'n6_base_t4', 'n6_scramble_t4', 'n6_heldout_t4')]
    run('rescore q17b B=1', base + ['--model', 'Qwen/Qwen3-1.7B-Base', '--files'] + files + ['--out', 'res/_rescore_q17b_b1.jsonl'], e2)
    for repo, name, q in [('Qwen/Qwen3-0.6B-Base', 'q06b', 'none'), ('Qwen/Qwen3-1.7B-Base', 'q17b', 'none'),
                          ('Qwen/Qwen3-1.7B', 'q17', 'none'), ('Qwen/Qwen3-4B-Base', 'q4b', 'nf4')]:
        run(f'ctrl_vocab {name}', [GPY, '-u', 'ctrl_vocab.py', '--model', repo, '--name', name, '--quant', q], e5)
    log('GPU CHECKS FINISHED')


if __name__ == '__main__':
    main()
