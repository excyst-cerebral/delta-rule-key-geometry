#!/bin/sh
cd "$(dirname "$0")/.."
{
for seed in 0 1 2 3 4; do
  for arch in lti sel; do
    echo "$arch ft none 0 $seed"; echo "$arch joint none 0 $seed"; echo "$arch replay none 0 $seed"
    echo "$arch ewc none 0 $seed"
  done
  for lam in 0.01 0.03; do echo "sel ft sparse $lam $seed"; done
  for lam in 0.5 2.0; do echo "sel ft decor $lam $seed"; done
done
} | xargs -P 2 -L 1 sh -c '
  arch=$0; method=$1; reg=$2; lam=$3; seed=$4; name="${arch}_${method}_${reg}${lam}_s${seed}"
  [ -f exp_cl/runs/${name}.json ] || ./.venv/bin/python -I exp_cl/cl_ssm.py --arch $arch --method $method --reg $reg --lam $lam --seed $seed --steps 400 --threads 1 --out exp_cl/runs/${name} >> exp_cl/runs.log 2>&1'
echo ALLDONE >> exp_cl/runs.log
