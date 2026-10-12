#!/bin/bash
# runs after master.sh: exploratory reasoning probe (Experiment 4)
T="${MASTER_OUT:?set MASTER_OUT to the file holding the output of master.sh}"
until grep -q "MASTER FINISHED" $T 2>/dev/null; do sleep 60; done
cd "$(dirname "$0")/exp4"
echo "START exp4:cot $(date +%H:%M:%S)"
/c/Users/$USERNAME/AppData/Local/Temp/cgpu/Scripts/python.exe -u cot.py --out res/qwen3-1.7b_cot.jsonl --n_per_cell 30 --max_new 1536 2>&1 | grep --line-buffered -E "done|DONE|Error|Traceback"
echo "END exp4:cot $(date +%H:%M:%S)"
