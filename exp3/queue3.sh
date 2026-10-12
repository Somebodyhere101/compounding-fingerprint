#!/bin/bash
# Experiment 3 training queue (runs alongside the exp2 LLM queue; each job is resumable)
GPY=/c/Users/$USERNAME/AppData/Local/Temp/cgpu/Scripts/python.exe
cd "$(dirname "$0")"
C="--orders 4 5 6 7 8 10 --steps 14000"
M="--env_mix group=0.5,latin=0.25,loop=0.25"
run() {  # name, minutes cap, args...
  name=$1; cap=$2; shift 2
  echo "TRAIN START $name $(date +%H:%M:%S)"
  $GPY -u train.py --name $name "$@" $C --minutes $cap 2>&1 | grep --line-buffered -E "DONE|Error|error|Traceback|OutOfMemory|nan"
  echo "TRAIN END $name $(date +%H:%M:%S)"
  $GPY -u eval_fp.py --ckpt runs/$name/final.pt --name $name --draws 4 2>&1 | grep --line-buffered -E "DONE|done|wrote|Error|Traceback" | tail -2
  echo "EVAL END $name $(date +%H:%M:%S)"
}
run A4 240 --layers 4 --types noheld $M
run I4 240 --layers 4 --env_mix isocopy=0.5,latin=0.5
run P4 240 --layers 4 --types noheld --env_mix group=1 --pool_M 32
run A1 120 --layers 1 --lr 1e-3 --types noheld $M
run A2 150 --layers 2 --lr 1e-3 --types noheld $M
run G4 240 --layers 4 --types noheld --env_mix group=1
run A6 360 --layers 6 --types noheld $M
/c/Users/$USERNAME/AppData/Local/Temp/ccv/Scripts/python.exe analyze3.py > /dev/null 2>&1
echo "EXP3 QUEUE FINISHED $(date +%H:%M:%S)"
