"""Validate the idealised-learner result files against c10 section 5.4 and run the preregistered decision rule
(imported from exp1/analyze.py) on each of them.

usage: python validate.py   -> prints tables; writes ../validation.json and ../validation_tables.txt
"""
import os, sys, json, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP1 = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, EXP1)
import analyze  # noqa: E402  (load, pooled, decision: the preregistered rule, unchanged)

LEARNERS = ['R', 'N', 'N_iso', 'A', 'Asl', 'B', 'Bh']
CLS = ['L', 'O', 'G', 'Ab', 'Cy', 'U']
RES = os.path.join(HERE, '..', 'res')
EXPECT = {'R': 'R', 'N': 'N', 'N_iso': 'N_iso', 'A': 'A', 'Asl': 'A', 'B': 'B-type', 'Bh': 'B-type'}

# c10 section 5.4 / 4.6 / 4.7 / 4.9 idealised predictions (G-cell accuracy unless noted)
C10 = {
    'R':     {'n6_base_t0': .17, 'n6_base_t1': .17, 'n6_base_t4': .17, 'n6_scramble_t4': .17, 'n6_heldout_t4': .17, 'n5_loops_t4': .20},
    'N':     {'n6_base_t0': .30, 'n6_base_t1': .30, 'n6_base_t4': .30, 'n6_scramble_t4': .30, 'n6_heldout_t4': .30,
              'n8_base_t0': .22, 'n8_base_t4': .22, 'n5_base_t0': .33, 'n5_loops_t4': .33},
    'N_iso': {'n6_base_t0': .30, 'n6_base_t1': .65, 'n6_base_t2': .83, 'n6_base_t4': .96, 'n6_scramble_t4': .30, 'n6_heldout_t4': .31,
              'n8_base_t0': .22, 'n8_base_t1': .38, 'n8_base_t4': .69, 'n8_heldout_t4': .22, 'n5_base_t4': 1.0, 'n5_loops_t4': .23},
    'A':     {'n6_base_t0': .30, 'n6_base_t1': 1.0, 'n6_base_t4': 1.0, 'n6_scramble_t4': .30, 'n6_heldout_t4': 1.0,
              'n8_base_t0': .22, 'n8_base_t1': 1.0, 'n8_base_t4': 1.0, 'n8_heldout_t4': 1.0, 'n5_base_t4': 1.0, 'n5_loops_t4': .46},
    'Asl':   {'n6_base_t0': .30, 'n6_base_t1': 1.0, 'n6_base_t4': 1.0, 'n6_scramble_t4': .30, 'n6_heldout_t4': 1.0, 'n5_loops_t4': 1.0},
    'B':     {'n6_base_t0': 1.0, 'n6_base_t1': 1.0, 'n6_base_t4': 1.0, 'n6_scramble_t4': 1.0, 'n6_heldout_t4': 1.0, 'n5_loops_t4': 1.0},
    'Bh':    {'n6_base_t0': 1.0, 'n6_base_t1': 1.0, 'n6_base_t4': 1.0, 'n6_heldout_t4': 1.0, 'n5_loops_t4': .47},
}
C10_O = {'N': {'n5_loops_t4': .35}, 'N_iso': {'n5_loops_t4': .54}, 'A': {'n5_loops_t4': 1.0}, 'Asl': {'n5_loops_t4': 1.0},
         'Bh': {'n5_loops_t4': 1.0}, 'B': {'n5_loops_t4': 1.0}, 'R': {'n5_loops_t4': .20}}


def per_class(path):
    acc = collections.defaultdict(lambda: [0, 0, 0.0])
    for l in open(path, encoding='utf-8'):
        r = json.loads(l)
        cell = r['file'].replace('.jsonl', '')
        for q in r['queries']:
            a = acc[(cell, q['class'])]
            a[0] += q['correct']; a[1] += 1; a[2] += q['p_gold']
            a = acc[(cell, 'all')]
            a[0] += q['correct']; a[1] += 1; a[2] += q['p_gold']
    return {k: (v[0] / v[1], v[2] / v[1], v[1]) for k, v in acc.items()}


def main():
    out = {'per_class': {}, 'decision': {}, 'c10_compare': {}}
    lines = []
    cells = sorted({json.loads(l)['file'].replace('.jsonl', '') for l in open(os.path.join(RES, 'ideal_R.jsonl'))},
                   key=lambda c: (int(c.split('_')[0][1:]), c))
    tabs = {L: per_class(os.path.join(RES, f'ideal_{L}.jsonl')) for L in LEARNERS}
    for cls in ['G', 'O', 'L', 'U', 'all']:
        lines.append(f'\n### accuracy on {cls} cells (sampled tie-breaks; mean p_gold in brackets)\n')
        lines.append('| cell | ' + ' | '.join(LEARNERS) + ' | cells |')
        lines.append('|---' * (len(LEARNERS) + 2) + '|')
        for c in cells:
            if (c, cls) not in tabs['R']:
                continue
            row = [c] + [f'{tabs[L][(c, cls)][0]:.3f} [{tabs[L][(c, cls)][1]:.3f}]' for L in LEARNERS] + [str(tabs['R'][(c, cls)][2])]
            lines.append('| ' + ' | '.join(row) + ' |')
    for L in LEARNERS:
        out['per_class'][L] = {f'{c}|{k}': v for (c, k), v in tabs[L].items()}
    # c10 comparison
    lines.append('\n### comparison with c10 idealised predictions (G cells; O for loops marked)\n')
    lines.append('| learner | cell | c10 | here acc | here mean p_gold | |diff| (p_gold) |')
    lines.append('|---|---|---|---|---|---|')
    cmp = []
    for L in LEARNERS:
        for src, cls in ((C10, 'G'), (C10_O, 'O')):
            for c, v in src.get(L, {}).items():
                if (c, cls) not in tabs[L]:
                    continue
                a, p, m = tabs[L][(c, cls)]
                cmp.append(dict(learner=L, cell=c, cls=cls, c10=v, acc=a, pgold=p, diff=abs(p - v)))
                lines.append(f'| {L} | {c} ({cls}) | {v:.2f} | {a:.3f} | {p:.3f} | {abs(p - v):.3f} |')
    out['c10_compare'] = cmp
    # decision rule
    lines.append('\n### preregistered decision rule (analyze.decision) on each learner\n')
    lines.append('| learner | expected | n | field | class | acc_L | g(G t0) | g(G scr4) | g(G t1) | g(G t4) | g(G held4) | comp [CI] | held [CI] | w | shape |')
    lines.append('|---' * 15 + '|')
    for L in LEARNERS:
        cl = analyze.load(os.path.join(RES, f'ideal_{L}.jsonl'))
        for prefix, n in (('n6', 6), ('n8', 8)):
            for field, lab in ((0, 'acc'), (2, 'p_gold')):
                s = analyze.decision(cl, n, prefix, field, boot=True)
                g = s['g']; ci = s['ci']
                key = f'{L}|{prefix}|{lab}'
                out['decision'][key] = {k: v for k, v in s.items()}
                ok = 'OK' if s['class'] == EXPECT[L] else 'MISMATCH'
                lines.append(f'| {L} | {EXPECT[L]} | {n} | {lab} | **{s["class"]}** ({ok}) | {s["accL"]:.3f} | '
                             f'{g[prefix + "_base_t0"]:.3f} | {g[prefix + "_scramble_t4"]:.3f} | {g[prefix + "_base_t1"]:.3f} | '
                             f'{g[prefix + "_base_t4"]:.3f} | {g[prefix + "_heldout_t4"]:.3f} | '
                             f'{s["comp"]:.3f} [{ci["comp"][0]:.3f},{ci["comp"][1]:.3f}] | {s["held"]:.3f} [{ci["held"][0]:.3f},{ci["held"][1]:.3f}] | '
                             f'{s["w"]:.2f} | {s["shape"]:.3f} |')
    # diagnostics
    diag = [json.loads(l) for l in open(os.path.join(RES, '_diag.jsonl'))]
    d = {
        'trials': len(diag),
        'class_mismatch_cells': sum(x['class_mismatch'] for x in diag),
        'sat_vs_exact_count_mismatch_cells': sum(x.get('sat_vs_count_mismatch', 0) for x in diag),
        'exact_latin_counts': sum(1 for x in diag if x.get('latin_N') is not None),
        'latin_capped': [(x['file'], x['trial']) for x in diag if x.get('latin_capped')],
        'Bh_fallback_possLatin': [(x['file'], x['trial']) for x in diag if x.get('Bh_fallback')],
        'Bh_latin_dropped_max_weight_bound': max([x.get('Bh_latin_dropped_wbound', 0.0) for x in diag]),
        'leak_meta_stale': sum(1 for x in diag if x.get('leak_meta_stale')),
        'max_latin_count_s': max([x.get('latin_count_s', 0.0) for x in diag]),
        'total_trial_secs': sum(x['secs'] for x in diag),
        'niso_match_rate_by_cell': {},
        'A_class_by_cell': {},
    }
    by = collections.defaultdict(list)
    for x in diag:
        by[x['file'].replace('.jsonl', '')].append(x)
    for c, xs in by.items():
        d['niso_match_rate_by_cell'][c] = float(np.mean([x['niso_match'] for x in xs]))
        d['A_class_by_cell'][c] = dict(collections.Counter(x['A_class'] for x in xs))
    out['diag'] = d
    lines.append('\n### diagnostics\n')
    lines.append(json.dumps(d, indent=1, default=str))
    txt = '\n'.join(lines)
    print(txt)
    open(os.path.join(HERE, '..', 'validation_tables.txt'), 'w', encoding='utf-8').write(txt + '\n')
    json.dump(out, open(os.path.join(HERE, '..', 'validation.json'), 'w'), default=float, indent=1)


if __name__ == '__main__':
    main()
