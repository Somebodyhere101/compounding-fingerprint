import json, glob, os, collections, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
for d in ['exp2/res', 'exp1/res']:
    for f in sorted(glob.glob(os.path.join(R, d, '*.jsonl'))):
        keysets = collections.Counter(); quant = collections.Counter(); off = collections.Counter(); B = collections.Counter(); bm = collections.Counter(); kv = collections.Counter()
        files = collections.Counter(); n = 0; dup = collections.Counter(); models = collections.Counter(); nq = 0
        order = []
        for l in open(f, encoding='utf-8'):
            r = json.loads(l); n += 1
            keysets[tuple(sorted(k for k in r if k != 'queries'))] += 1
            quant[r.get('quant')] += 1; off[r.get('offload_head', 'NA')] += 1; B[r.get('B_used', 'NA')] += 1
            bm[r.get('batch_max', 'NA')] += 1; kv[r.get('kv_budget_gb', 'NA')] += 1
            files[r['file']] += 1; dup[(r['file'], r['trial'])] += 1; models[r.get('model')] += 1; nq += len(r['queries'])
            order.append('new' if 'B_used' in r else 'old')
        # transitions
        trans = [i for i in range(1, len(order)) if order[i] != order[i-1]]
        print(os.path.relpath(f, R), 'n', n, 'nq', nq, 'models', dict(models), 'quant', dict(quant), 'offload', dict(off), 'batch_max', dict(bm), 'kv', dict(kv))
        print('   B_used', dict(sorted(B.items(), key=lambda x: str(x[0]))), 'dups', sum(1 for v in dup.values() if v > 1), 'old/new transitions at', trans[:5], 'nkeysets', len(keysets))
        print('   files', dict(files))
