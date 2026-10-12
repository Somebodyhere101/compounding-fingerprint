"""Idempotent finisher for every remaining experiment. Safe to re-run any time (after sleep, a crash, or a reboot).

It waits while any of the original queue scripts is still alive, then checks every expected artefact and runs only what
is missing or incomplete: LLM result files (resume at trial level), training runs (resume to 14,000 steps, then re-score),
Experiment 5 (CR axis), the Experiment 4 v2 probe, and finally the CPU analyses. Sleep only pauses it.

usage:  <ccv python> finish_all.py        (log: finish_all.log)
"""
import ctypes, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
GPY = r'C:\Users\%USERNAME%\AppData\Local\Temp\cgpu\Scripts\python.exe'
CPY = r'C:\Users\%USERNAME%\AppData\Local\Temp\ccv\Scripts\python.exe'
LOG = os.path.join(ROOT, 'finish_all.log')
QUEUE_PAT = r'master2\.sh|run_exp5\.sh|run_cot2\.sh|g4fix\.sh'
WORKER_PAT = r'run\.py --model|train\.py --name|eval_fp\.py --ckpt|cr_probe\.py --model|cot\.py --out'


def log(msg):
    line = f'{time.strftime("%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    open(LOG, 'a', encoding='utf-8').write(line + '\n')


def awake():  # hold off idle sleep while this runs (lid-close/manual sleep still pause everything; work resumes on wake)
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
    except Exception:
        pass


def n_alive(pattern):
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -ne 'powershell.exe' -and $_.ProcessId -ne %d -and "
          "$_.CommandLine -and $_.CommandLine -match '%s' } | Measure-Object | ForEach-Object { $_.Count }") % (os.getpid(), pattern)
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', ps], capture_output=True, text=True, timeout=120).stdout.strip()
        return int(out or 0)
    except Exception:
        return 1  # unknown -> assume busy


def run(label, cmd, cwd):
    """run a GPU job; retry transient code-integrity (Smart App Control) blocks up to 3 times."""
    for attempt in range(1, 4):
        awake(); log(f'START {label} (attempt {attempt})')
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        out = (p.stdout or '') + (p.stderr or '')
        tail = [l for l in out.splitlines() if any(k in l for k in ('DONE', 'Error', 'Traceback', 'calibration', 'resumed'))][-4:]
        for l in tail:
            log(f'  {l[:240]}')
        if any(k in out for k in ('Application Control', 'DLL load failed')):
            log(f'RETRY {label}: code-integrity block, waiting 2 min'); time.sleep(120); continue
        log(f'END {label} rc={p.returncode}')
        return p.returncode == 0
    log(f'GAVE UP {label}')
    return False


def nlines(p):
    return sum(1 for _ in open(p, encoding='utf-8')) if os.path.exists(p) else 0


def last_step(name):
    p = os.path.join(ROOT, 'exp3', 'runs', name, 'log.csv')
    if not os.path.exists(p):
        return 0
    rows = [l.split(',') for l in open(p, encoding='utf-8').read().splitlines()[1:] if l and l[0].isdigit()]
    return max((int(r[0]) for r in rows), default=0)


LLM = [  # repo, out name, quant, stim dir, expected lines, extra args
    ('Qwen/Qwen3-0.6B-Base', 'q06b_cal', 'none', 'stim', 1000, ['--mem_frac', '0.6']),
    ('Qwen/Qwen3-1.7B', 'q17_cal', 'none', 'stim', 1000, ['--mem_frac', '0.6']),
    ('HuggingFaceTB/SmolLM2-1.7B-Instruct', 'smol17_cal', 'none', 'stim', 1000, ['--mem_frac', '0.6']),
    ('Qwen/Qwen3-1.7B-Base', 'q17b_cal', 'none', 'stim', 1000, ['--mem_frac', '0.6']),
    ('Qwen/Qwen3-0.6B-Base', 'q06b_main', 'none', '../exp1/stim', 1800, ['--mem_frac', '0.6']),
    ('Qwen/Qwen3-1.7B-Base', 'q17b_main', 'none', '../exp1/stim', 1800, ['--mem_frac', '0.6']),
    ('Qwen/Qwen3-1.7B-Base', 'q17bnf4_main', 'nf4', '../exp1/stim', 1800, ['--mem_frac', '0.6']),
    ('Qwen/Qwen3-4B-Base', 'q4b_main', 'nf4', '../exp1/stim', 1800, ['--mem_frac', '0.6']),
    ('Qwen/Qwen3-8B-Base', 'q8b_main', 'nf4', '../exp1/stim', 1800, ['--offload_head', '--mem_frac', '0.6', '--kv_budget_gb', '0.5', '--batch', '16']),
    ('allenai/OLMo-2-0425-1B', 'olmo1b_main', 'none', '../exp1/stim', 1800, ['--mem_frac', '0.6']),
    ('allenai/OLMo-2-1124-7B', 'olmo7b_main', 'nf4', '../exp1/stim', 1800, ['--offload_head', '--mem_frac', '0.62', '--kv_budget_gb', '0.4', '--batch', '8']),
]
M3 = ['--env_mix', 'group=0.5,latin=0.25,loop=0.25']
TRAIN = {
    'A4': ['--layers', '4', '--types', 'noheld'] + M3, 'I4': ['--layers', '4', '--env_mix', 'isocopy=0.5,latin=0.5'],
    'P4': ['--layers', '4', '--types', 'noheld', '--env_mix', 'group=1', '--pool_M', '32'],
    'A1': ['--layers', '1', '--lr', '1e-3', '--types', 'noheld'] + M3, 'A2': ['--layers', '2', '--lr', '1e-3', '--types', 'noheld'] + M3,
    'G4': ['--layers', '4', '--types', 'noheld', '--env_mix', 'group=1'], 'A6': ['--layers', '6', '--types', 'noheld'] + M3,
}
CR = [('Qwen/Qwen3-0.6B-Base', 'q06b', ['--verify']), ('Qwen/Qwen3-1.7B-Base', 'q17b', []), ('Qwen/Qwen3-1.7B', 'q17', []),
      ('Qwen/Qwen3-4B-Base', 'q4b', ['--quant', 'nf4', '--max_q', '12']),
      ('Qwen/Qwen3-8B-Base', 'q8b', ['--quant', 'nf4', '--offload_head', '--max_q', '12'])]


def main():
    log('finisher started')
    while n_alive(QUEUE_PAT) > 0 or n_alive(WORKER_PAT) > 0:  # let the original queues finish first
        awake(); time.sleep(120)
    log('no queue running: checking artefacts')
    e2 = os.path.join(ROOT, 'exp2')
    for repo, name, quant, stim, exp, extra in LLM:
        have = nlines(os.path.join(e2, 'res', f'{name}.jsonl'))
        if have < exp:
            run(f'llm:{name} ({have}/{exp})', [GPY, '-u', 'run.py', '--model', repo, '--out', f'res/{name}.jsonl', '--quant', quant,
                                               '--batch', '32', '--kv_budget_gb', '1.5', '--stimdir', stim] + extra, e2)
    e3 = os.path.join(ROOT, 'exp3')
    for name, args in TRAIN.items():
        st = last_step(name)
        if st < 14000:
            fp = os.path.join(e3, 'runs', name, 'final.pt')
            if os.path.exists(fp):
                os.replace(fp, os.path.join(e3, 'runs', name, f'final_step{st}.pt'))
            rp = os.path.join(e3, 'res', f'{name}.jsonl')
            if os.path.exists(rp):
                os.replace(rp, os.path.join(e3, 'res', f'_{name}_step{st}.jsonl'))
            res = ['--resume'] if os.path.exists(os.path.join(e3, 'runs', name, 'last.pt')) else []
            run(f'train:{name} (from step {st})', [GPY, '-u', 'train.py', '--name', name] + args +
                ['--orders', '4', '5', '6', '7', '8', '10', '--steps', '14000'] + res, e3)
        if not os.path.exists(os.path.join(e3, 'res', f'{name}.jsonl')) and os.path.exists(os.path.join(e3, 'runs', name, 'final.pt')):
            run(f'eval:{name}', [GPY, '-u', 'eval_fp.py', '--ckpt', f'runs/{name}/final.pt', '--name', name, '--draws', '4'], e3)
    e5 = os.path.join(ROOT, 'exp5')
    for repo, name, extra in CR:
        if not os.path.exists(os.path.join(e5, 'res', f'{name}_meta.json')):
            run(f'cr:{name}', [GPY, '-u', 'cr_probe.py', '--model', repo, '--name', name] + extra, e5)
    e4 = os.path.join(ROOT, 'exp4')
    if nlines(os.path.join(e4, 'res', 'qwen3-1.7b_cot_v2.jsonl')) < 60:
        run('exp4:cot2', [GPY, '-u', 'cot.py', '--out', 'res/qwen3-1.7b_cot_v2.jsonl', '--n_per_cell', '15', '--max_new', '4096'], e4)
    for label, script, cwd in [('collect', 'collect.py', 'paper'), ('analyze2', 'analyze2.py', 'exp2'),
                               ('analyze5', 'analyze5.py', 'exp5'), ('figures', 'figures.py', 'paper')]:
        p = subprocess.run([CPY, script], cwd=os.path.join(ROOT, cwd), capture_output=True, text=True, encoding='utf-8', errors='replace',
                           env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        log(f'analysis {label} rc={p.returncode}')
    log('ALL FINISHED')


if __name__ == '__main__':
    main()
