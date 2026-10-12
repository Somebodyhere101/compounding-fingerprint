import time, sys
from huggingface_hub import snapshot_download
repos = ['Qwen/Qwen3-0.6B-Base', 'Qwen/Qwen3-1.7B-Base', 'Qwen/Qwen3-4B-Base', 'Qwen/Qwen3-8B-Base', 'allenai/OLMo-2-0425-1B', 'allenai/OLMo-2-1124-7B']
for r in repos:
    t = time.time()
    p = snapshot_download(r, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*', 'merges.txt', 'vocab.json'])
    print(f'DOWNLOADED {r} in {time.time()-t:.0f}s -> {p}', flush=True)
print('ALL DOWNLOADS DONE', flush=True)
