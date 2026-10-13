#!/bin/sh
# Task-specific adapters (task id given at test time).  Output: exp_bench/runs_adapter/
cd "$(dirname "$0")/.."
mkdir -p exp_bench/runs_adapter
{
for b in split perm; do for seed in 0 1 2 3 4; do for r in 0 1 4 16; do echo "$b $seed $r"; done; done; done
} | xargs -P 10 -L 1 sh -c '
  if [ $0 = split ]; then base="--lr 0.03 --epochs 4 --C 64"; else base="--lr 0.1 --epochs 3 --C 32 --ntrain 20000"; fi
  name="${0}_adapter_r${2}_s${1}"
  [ -f exp_bench/runs_adapter/${name}.json ] || ./.venv/bin/python -I exp_bench/cl_mnist.py --bench $0 --method adapter --rank $2 --seed $1 $base --threads 1 --out exp_bench/runs_adapter/${name} >> exp_bench/adapter.log 2>&1'
echo ALLDONE >> exp_bench/adapter.log
