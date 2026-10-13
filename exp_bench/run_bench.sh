#!/bin/sh
# Full grid for Section 9.7.  Hyper-parameters of the base training were tuned on fine-tuning only (seed 0), see exp_bench/tune/.
cd "$(dirname "$0")/.."
mkdir -p exp_bench/runs
{
for bench in split perm; do
  if [ $bench = split ]; then base="--lr 0.03 --epochs 4 --C 64"; else base="--lr 0.1 --epochs 3 --C 32 --ntrain 20000"; fi
  for seed in 0 1 2 3 4; do
    echo "$bench $seed ft none 0 0 32 free \"$base\""
    for l in 10 100 1000; do echo "$bench $seed ewc none $l 0 32 free \"$base\""; done
    for k in 20 200; do echo "$bench $seed replay none 0 $k 32 free \"$base\""; done
    for rho in 0.95 0.99; do echo "$bench $seed gpm $rho 0 0 32 free \"$base\""; done
    for rho in 0.95 0.99 0.999; do echo "$bench $seed gpm $rho 0 0 32 freeze \"$base\""; done
    for rho in 0.9 0.99; do echo "$bench $seed ogd $rho 0 0 32 free \"$base\""; done
  done
done
} | xargs -P 9 -L 1 sh -c '
  bench=$0; seed=$1; method=$2; rho=$3; lam=$4; k=$5; q=$6; ssm=$7; base=$8
  name="${bench}_${method}_rho${rho}_ewc${lam}_k${k}_q${q}_${ssm}_s${seed}"
  r=$rho; [ "$rho" = none ] && r=0.99
  l=$lam; [ "$lam" = 0 ] && l=100
  kk=$k; [ "$k" = 0 ] && kk=50
  [ -f exp_bench/runs/${name}.json ] || ./.venv/bin/python -I exp_bench/cl_mnist.py --bench $bench --method $method --seed $seed --rho $r --ewc $l --k $kk --q $q --ssm $ssm $base --threads 1 --out exp_bench/runs/${name} >> exp_bench/bench.log 2>&1'
echo ALLDONE >> exp_bench/bench.log
