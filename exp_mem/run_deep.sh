#!/bin/sh
cd "$(dirname "$0")/.."
{
for seed in 0 1 2 3 4; do
  echo "ft none 0 32 $seed"
  for rho in 0.5 0.9 0.99; do echo "ogd $rho 0 32 $seed"; done
  for q in 8 4 2; do echo "ogd 0.9 0 $q $seed"; done
  for lam in 1 10 100; do echo "ewc 0.9 $lam 32 $seed"; done
  for k in 4 32; do echo "replay 0.9 0 32 $seed $k"; done
done
} | xargs -P 5 -L 1 sh -c '
  method=$0; rho=$1; lam=$2; q=$3; seed=$4; k=${5:-16}; name="${method}_rho${rho}_ewc${lam}_q${q}_k${k}_s${seed}"
  [ -f exp_mem/runs/${name}.json ] || ./.venv/bin/python -I exp_mem/deep_ssm_cl.py --method $method --rho $rho --ewc $lam --q $q --k $k --seed $seed --threads 1 --out exp_mem/runs/${name} >> exp_mem/deep.log 2>&1'
echo ALLDONE >> exp_mem/deep.log
