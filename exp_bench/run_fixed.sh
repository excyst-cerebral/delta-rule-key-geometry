#!/bin/sh
# Fixed (never trained) random poles: the recurrence is a fixed bank of time scales, only linear layers are trained.  Toy streams + MNIST.
cd "$(dirname "$0")/.."
mkdir -p exp_mem/runs3 exp_bench/runs_fixed
{
for seed in 0 1 2 3 4; do
  for C in 12 32; do
    echo "toy $seed ft 0.99 $C"
    for rho in 0.99 0.999; do echo "toy $seed gpm $rho $C"; done
  done
  for b in split perm; do
    echo "$b $seed ft 0.99 0"; echo "$b $seed joint 0.99 0"
    for rho in 0.95 0.99 0.999; do echo "$b $seed gpm $rho 0"; done
  done
done
} | xargs -P 10 -L 1 sh -c '
  kind=$0; seed=$1; method=$2; rho=$3; C=$4
  if [ $kind = toy ]; then
    name="toy_${method}_rho${rho}_C${C}_s${seed}"
    [ -f exp_mem/runs3/${name}.json ] || ./.venv/bin/python -I exp_mem/deep_ssm_cl.py --method $method --ssm fixed --rho $rho --C $C --seed $seed --threads 1 --out exp_mem/runs3/${name} >> exp_bench/fixed.log 2>&1
  else
    if [ $kind = split ]; then base="--lr 0.03 --epochs 4 --C 64"; else base="--lr 0.1 --epochs 3 --C 32 --ntrain 20000"; fi
    name="${kind}_${method}_rho${rho}_s${seed}"
    [ -f exp_bench/runs_fixed/${name}.json ] || ./.venv/bin/python -I exp_bench/cl_mnist.py --bench $kind --method $method --ssm fixed --rho $rho --seed $seed $base --threads 1 --out exp_bench/runs_fixed/${name} >> exp_bench/fixed.log 2>&1
  fi'
echo ALLDONE >> exp_bench/fixed.log
