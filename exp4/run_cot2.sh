#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; GPY=/c/Users/$USERNAME/AppData/Local/Temp/cgpu/Scripts/python.exe; LOG="$ROOT/exp4/cot2.log"
until grep -q "EXP5 FINISHED" "$ROOT/exp5/exp5.log" 2>/dev/null; do sleep 60; done
/c/Users/$USERNAME/AppData/Local/Temp/ccv/Scripts/python.exe "$ROOT/keepawake.py" "$LOG" "END exp4:cot2" &
cd "$ROOT/exp4"; echo "START exp4:cot2 $(date +%H:%M:%S)" >> "$LOG"
"$GPY" -u cot.py --out res/qwen3-1.7b_cot_v2.jsonl --n_per_cell 15 --max_new 4096 >> "$LOG" 2>&1
echo "END exp4:cot2 $(date +%H:%M:%S)" >> "$LOG"
