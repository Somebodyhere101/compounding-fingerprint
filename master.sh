#!/bin/bash
# Single sequential GPU scheduler (no concurrency: the 8 GB laptop GPU is shared with other applications).
GPY=/c/Users/$USERNAME/AppData/Local/Temp/cgpu/Scripts/python.exe
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOCK=/c/Users/$USERNAME/AppData/Local/Temp/cthk/GPU_BUSY
F="grep --line-buffered -E 'loaded|DONE|Error|error|Traceback|OutOfMemory|assert|nan'"
llm() {  # repo outname quant stim extra...
  repo=$1; name=$2; quant=$3; stim=$4; shift 4
  echo "START llm:$name $(date +%H:%M:%S)"
  cd "$ROOT/exp2"; src="--stimdir $stim"
  "$GPY" -u run.py --model "$repo" --out "res/$name.jsonl" --quant "$quant" --batch 32 --kv_budget_gb 1.5 $src "$@" 2>&1 | eval $F
  echo "END llm:$name $(date +%H:%M:%S)"
}
C3="--orders 4 5 6 7 8 10 --steps 14000"
M3="--env_mix group=0.5,latin=0.25,loop=0.25"
tr3() {  # name cap args...
  name=$1; cap=$2; shift 2
  echo "START train:$name $(date +%H:%M:%S)"
  cd "$ROOT/exp3"; res=""; [ -f "runs/$name/last.pt" ] && res="--resume"
  [ -f "runs/$name/final.pt" ] || "$GPY" -u train.py --name $name "$@" $C3 --minutes $cap $res 2>&1 | eval $F
  echo "END train:$name $(date +%H:%M:%S)"
  [ -f "res/$name.jsonl" ] || "$GPY" -u eval_fp.py --ckpt runs/$name/final.pt --name $name --draws 4 2>&1 | grep --line-buffered -E "Error|Traceback" 
  echo "EVALED train:$name $(date +%H:%M:%S)"
}
touch $LOCK
llm Qwen/Qwen3-4B-Base q4b_main nf4 ../exp1/stim --mem_frac 0.6
tr3 A4 150 --layers 4 --types noheld $M3
llm Qwen/Qwen3-8B-Base q8b_cal nf4 stim --offload_head --mem_frac 0.6 --kv_budget_gb 0.5 --batch 16
llm Qwen/Qwen3-8B-Base q8b_main nf4 ../exp1/stim --offload_head --mem_frac 0.6 --kv_budget_gb 0.5 --batch 16
tr3 I4 150 --layers 4 --env_mix isocopy=0.5,latin=0.5
llm allenai/OLMo-2-1124-7B olmo7b_cal nf4 stim --offload_head --mem_frac 0.6 --kv_budget_gb 0.4 --batch 16
llm allenai/OLMo-2-1124-7B olmo7b_main nf4 ../exp1/stim --offload_head --mem_frac 0.6 --kv_budget_gb 0.4 --batch 16
tr3 P4 150 --layers 4 --types noheld --env_mix group=1 --pool_M 32
tr3 A1 60 --layers 1 --lr 1e-3 --types noheld $M3
tr3 A2 90 --layers 2 --lr 1e-3 --types noheld $M3
llm allenai/OLMo-2-0425-1B olmo1b_cal none stim --mem_frac 0.6
llm allenai/OLMo-2-0425-1B olmo1b_main none ../exp1/stim --mem_frac 0.6
llm Qwen/Qwen3-1.7B-Base q17b_main none ../exp1/stim --mem_frac 0.6
tr3 G4 150 --layers 4 --types noheld --env_mix group=1
tr3 A6 240 --layers 6 --types noheld $M3
llm Qwen/Qwen3-1.7B-Base q17bnf4_main nf4 ../exp1/stim --mem_frac 0.6
llm Qwen/Qwen3-1.7B-Base q17b_cal none stim --mem_frac 0.6
llm Qwen/Qwen3-0.6B-Base q06b_cal none stim --mem_frac 0.6
llm Qwen/Qwen3-1.7B q17_cal none stim --mem_frac 0.6
llm HuggingFaceTB/SmolLM2-1.7B-Instruct smol17_cal none stim --mem_frac 0.6
rm -f $LOCK
echo "MASTER FINISHED $(date +%H:%M:%S)"
