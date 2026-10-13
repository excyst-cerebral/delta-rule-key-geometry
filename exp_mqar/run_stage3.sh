#!/bin/sh
# stage 3: seeds 6-11 for baseline and key decorrelation (paired), to power the main comparison
cd "$(dirname "$0")/.."
{ for seed in 6 7 8 9 10 11; do echo "none 0 $seed"; echo "dec 1 $seed"; done; } | xargs -P 6 -L 1 sh -c '
  reg=$0; lam=$1; seed=$2; name="${reg}_l${lam}_s${seed}"; [ "$reg" = none ] && name="base_s${seed}"
  [ -f exp_mqar/runs/${name}.json ] || ./.venv/bin/python -I exp_mqar/mqar_train.py --reg $reg --lam $lam --seed $seed --steps 3000 --threads 2 --out exp_mqar/runs/${name} > exp_mqar/runs/${name}.log 2>&1
  echo "finished $name"' >> exp_mqar/grid3.log 2>&1
echo STAGE3DONE >> exp_mqar/grid3.log
