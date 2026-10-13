#!/bin/sh
# stage 4 (exp_mqar/PROTOCOL_replication.md): replication seeds 12-23 for baseline / key decorrelation, then seeds 6-11 for Gram matching and 3-11 for rank matching; 5 parallel workers x 2 threads
cd "$(dirname "$0")/.."
{ for seed in 12 13 14 15 16 17 18 19 20 21 22 23; do echo "none 0 $seed"; echo "dec 1 $seed"; done
  for seed in 6 7 8 9 10 11; do echo "match 1 $seed"; done
  for seed in 3 4 5 6 7 8 9 10 11; do echo "rank 0.1 $seed"; done; } | xargs -P 5 -L 1 sh -c '
  reg=$0; lam=$1; seed=$2; name="${reg}_l${lam}_s${seed}"; [ "$reg" = none ] && name="base_s${seed}"
  [ -f exp_mqar/runs/${name}.json ] || ./.venv/bin/python -I exp_mqar/mqar_train.py --reg $reg --lam $lam --seed $seed --steps 3000 --threads 2 --out exp_mqar/runs/${name} > exp_mqar/runs/${name}.log 2>&1
  echo "finished $name"' >> exp_mqar/grid4.log 2>&1
echo STAGE4DONE >> exp_mqar/grid4.log
