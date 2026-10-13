#!/bin/sh
# stage 1: baseline + one setting per regulariser, 3 seeds (base_s0 already done); 5 parallel workers x 2 threads
cd "$(dirname "$0")/.."
{
for seed in 0 1 2; do
  echo "none 0 $seed"; echo "dec 1 $seed"; echo "match 1 $seed"; echo "rank 0.1 $seed"
done
} | grep -v "^none 0 0$" | xargs -P 5 -L 1 sh -c '
  reg=$0; lam=$1; seed=$2; name="${reg}_l${lam}_s${seed}"; [ "$reg" = none ] && name="base_s${seed}"
  [ -f exp_mqar/runs/${name}.json ] || ./.venv/bin/python -I exp_mqar/mqar_train.py --reg $reg --lam $lam --seed $seed --steps 3000 --threads 2 --out exp_mqar/runs/${name} > exp_mqar/runs/${name}.log 2>&1
  echo "finished $name"'
