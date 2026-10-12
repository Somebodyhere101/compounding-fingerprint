#!/bin/bash
# Complete G4 to the preregistered 14,000 steps (it stopped at 9,934 when its wall-clock cap fired during overnight standby), then re-score it.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; GPY=/c/Users/$USERNAME/AppData/Local/Temp/cgpu/Scripts/python.exe; LOG="$ROOT/exp3/g4fix.log"
until grep -q "END exp4:cot2" "$ROOT/exp4/cot2.log" 2>/dev/null; do sleep 60; done
/c/Users/$USERNAME/AppData/Local/Temp/ccv/Scripts/python.exe "$ROOT/keepawake.py" "$LOG" "G4FIX FINISHED" &
cd "$ROOT/exp3"
mv runs/G4/final.pt runs/G4/final_step9934.pt; mv res/G4.jsonl res/_G4_step9934.jsonl
echo "START G4 resume $(date +%H:%M:%S)" >> "$LOG"
"$GPY" -u train.py --name G4 --layers 4 --types noheld --env_mix group=1 --orders 4 5 6 7 8 10 --steps 14000 --minutes 90 --resume 2>&1 | grep -E "resumed|DONE|Error|Traceback" >> "$LOG"
"$GPY" -u eval_fp.py --ckpt runs/G4/final.pt --name G4 --draws 4 2>&1 | grep -E "DONE|Error|Traceback" >> "$LOG"
echo "G4FIX FINISHED $(date +%H:%M:%S)" >> "$LOG"
