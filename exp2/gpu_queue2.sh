#!/bin/bash
# usage: bash gpu_queue.sh jobs.txt   (each line: <hf repo> <out name> <quant> <stimdir or files...>)
LOCK=/c/Users/$USERNAME/AppData/Local/Temp/cthk/GPU_BUSY
GPY=/c/Users/$USERNAME/AppData/Local/Temp/cgpu/Scripts/python.exe
cd "$(dirname "$0")"
while read -r repo name quant stim extra; do
  [ -z "$repo" ] && continue; [[ "$repo" == \#* ]] && continue
  touch $LOCK
  echo "START $name $(date +%H:%M:%S)"
  if [ -d "$stim" ]; then src="--stimdir $stim"; else src="--files $stim"; fi
  "$GPY" -u run.py --model "$repo" --out "res/$name.jsonl" --quant "$quant" --batch 32 --kv_budget_gb 1.5 $src $extra 2>&1 | grep --line-buffered -E "loaded|DONE|Error|error|Traceback|OutOfMemory|assert"
  echo "END $name $(date +%H:%M:%S)"
  rm -f $LOCK
done < "$1"
echo "QUEUE FINISHED $(date +%H:%M:%S)"
