"""Replicates the upstream eval protocol exactly (tester.get_test_batches: k_shots=200, hold_out=1, unshuffled=True, i.e.
vocabulary '0123...' assigned in element order, identity of the first group = '0'; accuracy = argmax over the full
vocabulary at the final '=' position). Expected: >= 0.99 (the checkpoint is the best-eval-accuracy snapshot)."""
import random, json, numpy as np, torch, ica
model = ica.load()
prng = random.Random(7); corr = []
for s in range(0, 1024, 64):
    xs = []
    for _ in range(64):
        groups, total = [], 0
        while True:
            opts = ica.rosette_groups(min(10, ica.NSYM - total))
            if not opts: break
            g = prng.choice(opts); groups.append(g); total += ica.table(*g).shape[0]
            if prng.random() > 0.7: break
        elems = [(gi, e) for gi, g in enumerate(groups) for e in range(ica.table(*g).shape[0])]
        w = {el: i for i, el in enumerate(elems)}
        facts = [((gi, a), (gi, b)) for gi, g in enumerate(groups) for a in range(ica.table(*g).shape[0]) for b in range(ica.table(*g).shape[0])]
        held = facts.pop(prng.randrange(len(facts)))
        if (held[1], held[0]) in facts: facts.remove((held[1], held[0]))
        toks = []
        for (a, b) in [prng.choice(facts) for _ in range(199)] + [held]:
            c = (a[0], int(ica.table(*groups[a[0]])[a[1], b[1]])); toks += [ica.SEP, w[a], w[b], ica.EQ, w[c]]
        xs.append(toks)
    x = torch.tensor(xs, device='cuda')
    with torch.no_grad():
        p = model(x[:, :-1])[:, -1].float().argmax(-1)
    corr += (p == x[:, -1]).cpu().tolist()
print('unshuffled k=200 final-token accuracy', np.mean(corr), len(corr))
json.dump({'unshuffled_k200_acc': float(np.mean(corr)), 'n': len(corr)}, open('validate_unshuffled.json', 'w'))
