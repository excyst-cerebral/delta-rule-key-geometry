#!/bin/sh
# wait for the stage-1 grid, then seeds 3-5 for baseline / decorrelation / matching
cd "$(dirname "$0")/.."
while pgrep -f run_grid.sh > /dev/null; do sleep 20; done
{ for seed in 3 4 5; do echo "none 0 $seed"; echo "dec 1 $seed"; echo "match 1 $seed"; done; } | xargs -P 5 -L 1 sh -c '
  reg=$0; lam=$1; seed=$2; name="${reg}_l${lam}_s${seed}"; [ "$reg" = none ] && name="base_s${seed}"
  [ -f exp_mqar/runs/${name}.json ] || ./.venv/bin/python -I exp_mqar/mqar_train.py --reg $reg --lam $lam --seed $seed --steps 3000 --threads 2 --out exp_mqar/runs/${name} > exp_mqar/runs/${name}.log 2>&1
  echo "finished $name"' >> exp_mqar/grid2.log 2>&1
echo STAGE2DONE >> exp_mqar/grid2.log
