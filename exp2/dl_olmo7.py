import time
from huggingface_hub import snapshot_download
for attempt in range(8):
    try:
        p = snapshot_download('allenai/OLMo-2-1124-7B', allow_patterns=['*.json', '*.safetensors', '*.txt', 'tokenizer*'])
        print('DOWNLOADED allenai/OLMo-2-1124-7B ->', p, flush=True); break
    except Exception as e:
        print(f'retry {attempt + 1}: {type(e).__name__}', flush=True); time.sleep(30)
else:
    print('DOWNLOAD FAILED allenai/OLMo-2-1124-7B', flush=True)
