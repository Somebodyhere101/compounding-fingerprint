#!/bin/bash
# After master.sh: (1) re-run OLMo-2-7B with the in-place single-query cache fix (resumes partial files), (2) Exp-4 reasoning probe.
T="${MASTER_OUT:?set MASTER_OUT to the file holding the output of master.sh}"
GPY=/c/Users/$USERNAME/AppData/Local/Temp/cgpu/Scripts/python.exe
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="$ROOT/exp4/after_master.log"
until grep -q "MASTER FINISHED" $T 2>/dev/null; do sleep 60; done
cd "$ROOT/exp2"
for job in "olmo7b_cal stim" "olmo7b_main ../exp1/stim"; do
  set -- $job
  echo "START llm:$1(retry) $(date +%H:%M:%S)" >> "$LOG"
  "$GPY" -u run.py --model allenai/OLMo-2-1124-7B --out res/$1.jsonl --quant nf4 --stimdir $2 --offload_head --mem_frac 0.62 --kv_budget_gb 0.4 --batch 8 2>&1 | grep --line-buffered -E "loaded|DONE|Error|Traceback|OutOfMemory" >> "$LOG"
  echo "END llm:$1(retry) $(date +%H:%M:%S)" >> "$LOG"
done
cd "$ROOT/exp4"
echo "START exp4:cot $(date +%H:%M:%S)" >> "$LOG"
"$GPY" -u cot.py --out res/qwen3-1.7b_cot.jsonl --n_per_cell 30 --max_new 1536 2>&1 | grep --line-buffered -E "done|DONE|Error|Traceback" >> "$LOG"
echo "END exp4:cot $(date +%H:%M:%S)" >> "$LOG"
