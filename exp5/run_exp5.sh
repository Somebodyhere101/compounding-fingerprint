#!/bin/bash
# Experiment 5 queue: starts after master2.sh finishes (GPU is free), holds the machine awake while it runs.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; GPY=/c/Users/$USERNAME/AppData/Local/Temp/cgpu/Scripts/python.exe
LOG="$ROOT/exp5/exp5.log"
until grep -q "MASTER2 FINISHED" "$ROOT/master2.log" 2>/dev/null; do sleep 60; done
/c/Users/$USERNAME/AppData/Local/Temp/ccv/Scripts/python.exe "$ROOT/keepawake.py" "$LOG" "EXP5 FINISHED" &
cd "$ROOT/exp5"
job() {  # model name args...
  m=$1; n=$2; shift 2
  for a in 1 2 3; do
    echo "START cr:$n attempt $a $(date +%H:%M:%S)" >> "$LOG"
    out=$("$GPY" -u cr_probe.py --model "$m" --name "$n" "$@" 2>&1); echo "$out" | grep -E "verify|calibration|DONE|Error|Traceback" >> "$LOG"
    echo "$out" | grep -qE "ImportError|Application Control|DLL load failed" && { sleep 120; continue; }
    break
  done
  echo "END cr:$n $(date +%H:%M:%S)" >> "$LOG"
}
job Qwen/Qwen3-0.6B-Base q06b --verify
job Qwen/Qwen3-1.7B-Base q17b
job Qwen/Qwen3-1.7B q17
job Qwen/Qwen3-4B-Base q4b --quant nf4 --max_q 12
job Qwen/Qwen3-8B-Base q8b --quant nf4 --offload_head --max_q 12
echo "EXP5 FINISHED $(date +%H:%M:%S)" >> "$LOG"
