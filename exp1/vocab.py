"""Build the shared vocabulary: words that are exactly one token (with and without a leading space)
in BOTH the Qwen3 and SmolLM2 (Llama-architecture) tokenizers. Uses c10's DEFAULT_WORDS and filter rules."""
import os, re, sys
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import AutoTokenizer
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, '..', 'runs', 'c10', 'code', 'stimuli.py'), encoding='utf-8').read()
words = re.search(r"DEFAULT_WORDS = '''(.*?)'''", src, re.S).group(1).split()
black = set(re.search(r"BLACKLIST = set\('''(.*?)'''", src, re.S).group(1).split())
HUB = os.path.expanduser('~/.cache/huggingface/hub')
def snap(repo):
    d = os.path.join(HUB, 'models--' + repo.replace('/', '--'), 'snapshots')
    return os.path.join(d, os.listdir(d)[0])
toks = {r: AutoTokenizer.from_pretrained(snap(r), local_files_only=True) for r in ['Qwen/Qwen3-1.7B', 'HuggingFaceTB/SmolLM2-1.7B-Instruct']}
out, seen = [], set()
for w in words:
    w = w.strip().lower()
    if not w.isalpha() or w in black or w in seen or not (3 <= len(w) <= 8):
        continue
    ok = all(len(t.encode(' ' + w, add_special_tokens=False)) == 1 and len(t.encode(w, add_special_tokens=False)) == 1 for t in toks.values())
    if ok:
        seen.add(w); out.append(w)
open(os.path.join(here, 'vocab_common.txt'), 'w').write('\n'.join(out))
print('candidates', len(words), '-> common single-token vocab', len(out))
print(' '.join(out[:60]))
