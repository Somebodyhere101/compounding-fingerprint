"""Queue the addendum-2d k=16 scoring (Qwen3-0.6B-Base, Qwen3-1.7B-Base) after gpu_checks.py finishes. Resumable. Log: gpu_2d.log"""
import ctypes, os, subprocess, time
ROOT = os.path.dirname(os.path.abspath(__file__))
GPY = r'C:\Users\%USERNAME%\AppData\Local\Temp\cgpu\Scripts\python.exe'
CPY = r'C:\Users\%USERNAME%\AppData\Local\Temp\ccv\Scripts\python.exe'
LOG = os.path.join(ROOT, 'gpu_2d.log')
def log(m):
    open(LOG, 'a', encoding='utf-8').write(f'{time.strftime("%m-%d %H:%M:%S")} {m}\n')
def awake():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
while 'GPU CHECKS FINISHED' not in open(os.path.join(ROOT, 'gpu_checks.log'), encoding='utf-8').read():
    awake(); time.sleep(60)
for repo, name in [('Qwen/Qwen3-0.6B-Base', 'q06b'), ('Qwen/Qwen3-1.7B-Base', 'q17b')]:
    for attempt in range(3):
        awake(); log(f'START {name}_k16 attempt {attempt + 1}')
        p = subprocess.run([GPY, '-u', 'run.py', '--model', repo, '--out', f'res/{name}_k16.jsonl', '--stimdir', 'stim_k16', '--batch', '32',
                            '--kv_budget_gb', '1.5', '--mem_frac', '0.6'], cwd=os.path.join(ROOT, 'exp2'), capture_output=True, text=True, encoding='utf-8', errors='replace')
        out = (p.stdout or '') + (p.stderr or '')
        log(' | '.join(l for l in out.splitlines() if 'DONE' in l or 'Error' in l)[-300:] + f' rc={p.returncode}')
        if p.returncode == 0:
            break
        time.sleep(60)
p = subprocess.run([CPY, 'analyze_2d.py'], cwd=os.path.join(ROOT, 'exp2'), capture_output=True, text=True, encoding='utf-8', errors='replace', env=dict(os.environ, PYTHONIOENCODING='utf-8'))
log(f'analyze_2d rc={p.returncode}'); log('2D FINISHED')
