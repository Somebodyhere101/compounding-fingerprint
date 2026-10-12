#!/bin/bash
# Resumed sequential GPU scheduler (2026-10-08). Each job retries up to 3x on transient failures
# (e.g. code-integrity DLL blocks right after the laptop resumes from Modern Standby). All jobs are resumable.
GPY=/c/Users/$USERNAME/AppData/Local/Temp/cgpu/Scripts/python.exe
ROOT="$(cd "$(dirname "$0")" && pwd)"
F='loaded|DONE|Error|Traceback|OutOfMemory|assert|nan'
retry() {  # label, then the command; retries if the command output contains ImportError/Application Control
  label=$1; shift
  for a in 1 2 3; do
    out=$("$@" 2>&1); echo "$out" | grep -E "$F" | tail -4
    if echo "$out" | grep -qE "ImportError|Application Control|DLL load failed"; then echo "RETRY $label attempt $a $(date +%H:%M:%S)"; sleep 120; else return 0; fi
  done
  echo "GAVE UP $label"
}
llm() {  # repo outname quant stim extra...
  repo=$1; name=$2; quant=$3; stim=$4; shift 4
  echo "START llm:$name $(date +%H:%M:%S)"; cd "$ROOT/exp2"
  retry "llm:$name" "$GPY" -u run.py --model "$repo" --out "res/$name.jsonl" --quant "$quant" --batch 32 --kv_budget_gb 1.5 --stimdir "$stim" "$@"
  echo "END llm:$name $(date +%H:%M:%S)"
}
C3="--orders 4 5 6 7 8 10 --steps 14000"; M3="--env_mix group=0.5,latin=0.25,loop=0.25"
tr3() {  # name cap args...
  name=$1; cap=$2; shift 2
  echo "START train:$name $(date +%H:%M:%S)"; cd "$ROOT/exp3"
  res=""; [ -f "runs/$name/last.pt" ] && res="--resume"
  [ -f "runs/$name/final.pt" ] || retry "train:$name" "$GPY" -u train.py --name $name "$@" $C3 --minutes $cap $res
  echo "END train:$name $(date +%H:%M:%S)"
  [ -f "res/$name.jsonl" ] || retry "eval:$name" "$GPY" -u eval_fp.py --ckpt runs/$name/final.pt --name $name --draws 4
  echo "EVALED train:$name $(date +%H:%M:%S)"
}
llm Qwen/Qwen3-1.7B-Base q17b_main none ../exp1/stim --mem_frac 0.6
llm Qwen/Qwen3-1.7B-Base q17bnf4_main nf4 ../exp1/stim --mem_frac 0.6
llm allenai/OLMo-2-0425-1B olmo1b_cal none stim --mem_frac 0.6
llm allenai/OLMo-2-0425-1B olmo1b_main none ../exp1/stim --mem_frac 0.6
llm allenai/OLMo-2-1124-7B olmo7b_cal nf4 stim --offload_head --mem_frac 0.62 --kv_budget_gb 0.4 --batch 8
llm allenai/OLMo-2-1124-7B olmo7b_main nf4 ../exp1/stim --offload_head --mem_frac 0.62 --kv_budget_gb 0.4 --batch 8
echo "START exp4:cot $(date +%H:%M:%S)"; cd "$ROOT/exp4"
retry "exp4:cot" "$GPY" -u cot.py --out res/qwen3-1.7b_cot.jsonl --n_per_cell 30 --max_new 1536
echo "END exp4:cot $(date +%H:%M:%S)"
tr3 A2 90 --layers 2 --lr 1e-3 --types noheld $M3
tr3 G4 150 --layers 4 --types noheld --env_mix group=1
tr3 A6 240 --layers 6 --types noheld $M3
llm Qwen/Qwen3-1.7B-Base q17b_cal none stim --mem_frac 0.6
llm Qwen/Qwen3-0.6B-Base q06b_cal none stim --mem_frac 0.6
llm Qwen/Qwen3-1.7B q17_cal none stim --mem_frac 0.6
llm HuggingFaceTB/SmolLM2-1.7B-Instruct smol17_cal none stim --mem_frac 0.6
echo "MASTER2 FINISHED $(date +%H:%M:%S)"
