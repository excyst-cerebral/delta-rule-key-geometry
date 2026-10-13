#!/bin/sh
# Section 9.6 follow-up grid: layerwise GPM variants, width control, learning-rate (laziness) control.  Output: exp_mem/runs2/
cd "$(dirname "$0")/.."
mkdir -p exp_mem/runs2
{
for seed in 0 1 2 3 4; do
  for ssm in freeze free ogd; do for rho in 0.99 0.999; do echo "gpm $ssm $rho 32 12 0.03 600 $seed 16"; done; done
  echo "gpm freeze 0.99 8 12 0.03 600 $seed 16"
  echo "gpm freeze 0.99 4 12 0.03 600 $seed 16"
  for rho in 0.99 0.999; do echo "gpm freeze $rho 32 32 0.03 600 $seed 16"; done
  echo "ft free 0.9 32 32 0.03 600 $seed 16"
  echo "replay free 0.9 32 32 0.03 600 $seed 32"
  echo "ewc free 0.9 32 32 0.03 600 $seed 16"
  echo "ft free 0.9 32 12 0.005 3600 $seed 16"
  echo "ogd free 0.99 32 12 0.005 3600 $seed 16"
done
} | xargs -P 8 -L 1 sh -c '
  method=$0; ssm=$1; rho=$2; q=$3; C=$4; lr=$5; steps=$6; seed=$7; k=$8; name="${method}_${ssm}_rho${rho}_q${q}_C${C}_lr${lr}_k${k}_s${seed}"
  [ -f exp_mem/runs2/${name}.json ] || ./.venv/bin/python -I exp_mem/deep_ssm_cl.py --method $method --ssm $ssm --rho $rho --q $q --C $C --lr $lr --steps $steps --seed $seed --k $k --ewc 10 --threads 1 --out exp_mem/runs2/${name} >> exp_mem/deep4.log 2>&1'
echo ALLDONE >> exp_mem/deep4.log
