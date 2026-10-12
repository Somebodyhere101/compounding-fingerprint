"""Addendum 2d operating point (run 2026-10-09 after the final audit found that, under the literal 2a rule without the >=30-query
guard, Qwen3-0.6B-Base and Qwen3-1.7B-Base have k_L = 6 and are R at k=10). Cells exactly as 2d specifies: n=6, k=16, base t in
{0,1,4}, scramble t=4, heldout t in {1,4}; 100 trials per cell; new seeds; same c10 generator and vocabulary as Experiment 1.
usage: python make_stim_k16.py  -> stim_k16/n6_*.jsonl (cell names as in Experiment 1, so exp1/analyze.decision applies unchanged)"""
import glob, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, '..', 'runs', 'c10', 'code')
VOC = os.path.join(HERE, '..', 'exp1', 'vocab_common.txt')
CELLS = [('base', 0, 216000), ('base', 1, 216001), ('base', 4, 216004), ('scramble', 4, 216104), ('heldout', 1, 216201), ('heldout', 4, 216204)]
for cond, t, seed in CELLS:
    out = os.path.join(HERE, 'stim_k16', f'n6_{cond}_t{t}.jsonl')
    if not os.path.exists(out):
        subprocess.run([sys.executable, 'stimuli.py', '--n', '6', '--cond', cond, '--t', str(t), '--k', '16', '--trials', '100',
                        '--vocab', VOC, '--seed', str(seed), '--out', out], cwd=GEN, check=True)
old = set()
for f in glob.glob(os.path.join(HERE, '..', 'exp1', 'stim', '*.jsonl')) + glob.glob(os.path.join(HERE, 'stim', '*.jsonl')):
    old |= {json.loads(l)['context'] for l in open(f, encoding='utf-8')}
for f in sorted(glob.glob(os.path.join(HERE, 'stim_k16', '*.jsonl'))):
    rows = [json.loads(l) for l in open(f, encoding='utf-8')]
    assert len(rows) == 100 and all(r['k'] == 16 for r in rows)
    dup = sum(r['context'] in old for r in rows)
    cls = {}
    for r in rows:
        for q in r['queries']:
            cls[q['class']] = cls.get(q['class'], 0) + 1
    print(os.path.basename(f), 'trials', len(rows), 'reused contexts', dup, 'classes', cls, 'target types', sorted({tuple(r['target_type']) for r in rows}))
