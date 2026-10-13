#!/bin/bash
# waits for Part A, runs its analysis, then Part B extraction (5 shards per model) and analysis
cd .
while [ ! -f exp_gate/out/model_gdn340m_actual.npz ] || [ ! -f exp_gate/out/model_gdn360m_actual.npz ]; do sleep 20; done
sleep 5
for m in model_gdn340m model_gdn360m; do .venv/bin/python -I exp_gate/analyze_actual.py $m > exp_gate/out/$m.analysis_actual.log 2>&1; done
for m in model_gdn340m model_gdn360m; do for k in 0 1 2 3 4; do GDN_MD=$m NT=1 .venv/bin/python -I exp_gate/extract_gate_task.py $k 5 150 > exp_gate/out/$m.task$k.log 2>&1 & done; done
wait
.venv/bin/python -I exp_gate/analyze_task_gate.py > exp_gate/out/task_gate_analysis.log 2>&1
echo ALL_DONE > exp_gate/out/ALL_DONE
