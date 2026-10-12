"""Post-audit diagnostic (2026-10-09, exploratory): do trained nets / LMs answer with a single global 'least-revealed word' per trial?
For every scored trial: (a) within-trial prediction constancy = share of queries equal to the trial's modal prediction;
(b) agreement of each prediction with the count heuristic: candidates whose number of occurrences as a *result* in the revealed target
cells is minimal (ties -> set); (c) accuracy of the count heuristic itself (uniform over the tied set) and of the cell-level Latin
row/column elimination (uniform over values not yet in the cell's row and column). usage: python count_heuristic.py"""
import json, os, re, glob, zlib
from collections import Counter, defaultdict
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACT = re.compile(r'^(\w+) \* (\w+) = (\w+)$')
SETS = [('A4 cal', 'exp3/res_cal/A4_cal.jsonl', 'exp2/stim'), ('A4 main', 'exp3/res/A4.jsonl', 'exp1/stim'),
        ('A6 main', 'exp3/res/A6.jsonl', 'exp1/stim'), ('A1 main', 'exp3/res/A1.jsonl', 'exp1/stim'), ('A2 main', 'exp3/res/A2.jsonl', 'exp1/stim'), ('G4 main', 'exp3/res/G4.jsonl', 'exp1/stim'),
        ('I4 main', 'exp3/res/I4.jsonl', 'exp1/stim'), ('P4 main', 'exp3/res/P4.jsonl', 'exp1/stim'),
        ('Qwen3-8B-Base cal', 'exp2/res/q8b_cal.jsonl', 'exp2/stim'), ('Qwen3-1.7B-Base cal', 'exp2/res/q17b_cal.jsonl', 'exp2/stim'),
        ('Qwen3-8B-Base main', 'exp2/res/q8b_main.jsonl', 'exp1/stim')]
stim_cache = {}
def stim(d, f):
    if (d, f) not in stim_cache:
        stim_cache[(d, f)] = {r['trial']: r for r in map(json.loads, open(os.path.join(ROOT, d, f), encoding='utf-8'))}
    return stim_cache[(d, f)]
out = []
TD, TC = {}, {}
for lab, p, sd in SETS:
    if not os.path.exists(os.path.join(ROOT, p)):
        continue
    agg = defaultdict(lambda: defaultdict(list))
    for l in open(os.path.join(ROOT, p), encoding='utf-8'):
        r = json.loads(l)
        if r['file'].startswith(('ki_', 'kso_')):
            continue
        tr = stim(sd, r['file'])[r['trial']]
        target = [FACT.match(x).groups() for x in tr['context'].split('Table ')[-1].split('\n')[1:] if FACT.match(x)]
        cnt = Counter(c for _, _, c in target)
        words = tr['target_words']; mn = min(cnt.get(w, 0) for w in words); low = {w for w in words if cnt.get(w, 0) == mn}
        row = defaultdict(set); col = defaultdict(set)
        for a, b, c in target:
            row[a].add(c); col[b].add(c)
        preds = [q['pred'] for q in r['queries']]; modal = Counter(preds).most_common(1)[0][1] / len(preds)
        key = f"k{r.get('k', tr.get('k'))}" if sd == 'exp2/stim' else r['file'][:-6]
        A = agg[key]
        A['constancy'].append(modal)
        TD.setdefault(lab, []).append(np.mean([q['correct'] for q in r['queries']]) - np.mean([(q['gold'] in low) / len(low) for q in r['queries']]))
        TC.setdefault(lab, []).append(np.mean([q['pred'] in low for q in r['queries']]))
        for q in r['queries']:
            wa, wb = re.findall(r'\w+', q['prompt_suffix'] if 'prompt_suffix' in q else '')[:2] if 'prompt_suffix' in q else (None, None)
            qq = next(x for x in tr['queries'] if x['cell'] == q['cell'])
            wa, wb = re.findall(r'\w+', qq['prompt_suffix'])[:2]
            feas = [w for w in words if w not in row[wa] and w not in col[wb]]
            A['pred_in_lowcount'].append(q['pred'] in low)
            A['count_heur_acc'].append((q['gold'] in low) / len(low))
            A['rowcol_acc'].append((q['gold'] in feas) / max(1, len(feas)))
            A['model_acc'].append(q['correct'])
            A['n_low'].append(len(low))
    out.append(f'## {lab}')
    out.append('| cell | trials | within-trial constancy | pred ∈ least-revealed set | P(least-revealed set) uniform | model acc | count-heuristic acc | row/col-elimination acc |')
    out.append('|---|---|---|---|---|---|---|---|')
    for key in sorted(agg, key=lambda s: (len(s), s)):
        A = agg[key]
        base_rate = np.mean([nl / 6 if key.startswith('k') or 'n6' in key else np.nan for nl in A['n_low']])
        out.append(f"| {key} | {len(A['constancy'])} | {np.mean(A['constancy']):.2f} | {np.mean(A['pred_in_lowcount']):.2f} | {base_rate:.2f} | "
                   f"{np.mean(A['model_acc']):.3f} | {np.mean(A['count_heur_acc']):.3f} | {np.mean(A['rowcol_acc']):.3f} |")
    td = np.array(TD[lab]); rng = np.random.default_rng(zlib.crc32(lab.encode()))
    bs = [rng.choice(td, len(td)).mean() for _ in range(2000)]
    out.append(f"All {len(td)} trials (raw-format cells): model acc − count-heuristic acc = {td.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]; "
               f"share of predictions in the least-revealed set {np.mean(TC[lab]):.3f}.")
    out.append('')
txt = '\n'.join(out); print(txt)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'count_heuristic.md'), 'w', encoding='utf-8').write(
    '# Count-heuristic diagnostic (exploratory, post-audit; auto-generated by count_heuristic.py)\n\n'
    '"least-revealed set" = target words with the fewest occurrences as a result in the revealed cue. Column 5 is the share of the 6 words in that set (n=6 cells only), i.e. the chance level for column 4.\n\n' + txt + '\n')
