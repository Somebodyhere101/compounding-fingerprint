#!/bin/bash
# Runs ideal.py as 8 independent OS processes (multiprocessing.Pool is denied by the sandbox:
# DuplicateHandle -> WinError 5), then merges the part files into res/.
# usage (from exp1/ideal/code): bash run_shards.sh [nshards]
PY=/c/Users/$USERNAME/AppData/Local/Temp/ccv/Scripts/python.exe
NS=${1:-8}
export LAT_CAP=${LAT_CAP:-3e8}
mkdir -p ../logs
for i in $(seq 0 $((NS-1))); do
  $PY -u ideal.py --shard $i --nshards $NS --partdir ../parts > ../logs/shard$i.txt 2>&1 &
done
wait
$PY -u ideal.py --merge --partdir ../parts --outdir ../res
