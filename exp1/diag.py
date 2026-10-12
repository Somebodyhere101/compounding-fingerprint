"""Response-bias diagnostics: how often the model answers with a word from the query itself (copy-a, copy-b),
the cue's most frequent word, or the identity element, by forcing class. Joins results with stimuli."""
import glob, json, os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    stim = {}
    for f in glob.glob(os.path.join(HERE, 'stim', '*.jsonl')):
        for l in open(f, encoding='utf-8'):
            t = json.loads(l)
            last = t['context'].split('\n\n')[-1].split('\n')[1:]
            freq = Counter(w for line in last for w in line.replace('*', ' ').replace('=', ' ').split())
            stim[(os.path.basename(f), t['trial'])] = (t['target_words'], freq.most_common(1)[0][0])
    for path in sorted(glob.glob(os.path.join(HERE, 'res', '*.jsonl'))):
        model = os.path.basename(path)[:-6]
        if model.startswith('_'):
            continue
        agg = defaultdict(Counter)
        for l in open(path, encoding='utf-8'):
            r = json.loads(l)
            tw, top = stim[(r['file'], r['trial'])]
            for q in r['queries']:
                a, b = q['cell']
                key = (r['file'][:2], q['class'])
                c = agg[key]
                c['n'] += 1
                c['correct'] += q['correct']
                c['copy_a'] += q['pred'] == tw[a]
                c['copy_b'] += q['pred'] == tw[b]
                c['copy_any'] += q['pred'] in (tw[a], tw[b])
                c['gold_is_a_or_b'] += q['gold'] in (tw[a], tw[b])
                c['top_freq'] += q['pred'] == top
        print(f'\n{model}')
        print('  n  class  queries  acc   copy_a copy_b COPY_ANY (gold=a|b)  pred=most-frequent-cue-word')
        for (nn, cl), c in sorted(agg.items()):
            N = c['n']
            print(f"  {nn}  {cl:5} {N:7}  {c['correct'] / N:.3f}  {c['copy_a'] / N:.3f}  {c['copy_b'] / N:.3f}  {c['copy_any'] / N:.3f}  ({c['gold_is_a_or_b'] / N:.3f})      {c['top_freq'] / N:.3f}")


if __name__ == '__main__':
    main()
