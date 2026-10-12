"""Full-sequence training loss of the re-implemented model on fresh training-distribution sequences (k=200, no hold-out,
shuffled vocabulary, loss on every next token as in upstream trainer.py). Upstream progress/train_loss.png ends at ~0.93."""
import random, json, numpy as np, torch, torch.nn.functional as F, ica
model = ica.load(); prng = random.Random(11); losses = []
for s in range(8):
    x = torch.tensor([ica.sample_run(prng, k_shots=200, holdout_final=False)[0] for _ in range(32)], device='cuda')
    with torch.no_grad():
        lg = model(x[:, :-1]).float()
    losses.append(F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1)).item())
print('mean training loss', np.mean(losses), losses)
json.dump({'train_loss_reimpl': float(np.mean(losses)), 'upstream_plot_final': '~0.93'}, open('validate_loss.json', 'w'))
