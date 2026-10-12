import os
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import AutoTokenizer
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
HUB = os.path.expanduser('~/.cache/huggingface/hub')
words = [w.strip() for w in open(os.path.join(R, 'exp1', 'vocab_common.txt'), encoding='utf-8') if w.strip()]
print('nwords', len(words))
for repo in ['allenai/OLMo-2-0425-1B', 'allenai/OLMo-2-1124-7B', 'Qwen/Qwen3-8B-Base', 'Qwen/Qwen3-1.7B-Base', 'HuggingFaceTB/SmolLM2-1.7B-Instruct']:
    d = os.path.join(HUB, 'models--' + repo.replace('/', '--'), 'snapshots')
    if not os.path.isdir(d):
        print(repo, 'not cached'); continue
    tok = AutoTokenizer.from_pretrained(os.path.join(d, os.listdir(d)[0]), local_files_only=True)
    bad_sp = [w for w in words if len(tok.encode(' ' + w, add_special_tokens=False)) != 1]
    bad_ns = [w for w in words if len(tok.encode(w, add_special_tokens=False)) != 1]
    # in-context: after newline (start of a fact line)
    bad_nl = [w for w in words if len(tok.encode('\n' + w, add_special_tokens=False)) != len(tok.encode('\n', add_special_tokens=False)) + 1]
    print(repo, 'multi-token with space:', bad_sp[:10], len(bad_sp), '| without space:', bad_ns[:10], len(bad_ns), '| after newline:', len(bad_nl), 'bos:', tok.bos_token, tok('x')['input_ids'][:3])
