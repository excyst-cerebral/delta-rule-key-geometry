#!/bin/sh
# held-out runs (exp_heldout/PROTOCOL.md), sequentially and at low priority; then the analysis
cd "$(dirname "$0")/.."
PY=./.venv/bin/python; export HELDOUT=1
L=exp_heldout/out
nice -n 15 $PY -I rwkv7_flat.py run > $L/rwkv7.log 2>&1
nice -n 15 $PY -I tf_flat.py models--HuggingFaceTB--SmolLM2-135M-Instruct > $L/smollm2.log 2>&1
nice -n 15 $PY -I tf_flat.py models--Qwen--Qwen2.5-1.5B-Instruct > $L/qwen25.log 2>&1
nice -n 15 $PY -I qwen35_flat.py extract > $L/qwen35_extract.log 2>&1
nice -n 15 $PY -I qwen35_flat.py analyze > $L/qwen35_analyze.log 2>&1
GDN_MD=model_gdn360m nice -n 15 $PY -I gdn_transfer2.py > $L/gdn360m.log 2>&1
GDN_MD=model_gdn340m nice -n 15 $PY -I gdn_transfer2.py > $L/gdn340m.log 2>&1
$PY -I exp_heldout/analyze_heldout.py > $L/analysis.log 2>&1
echo DONE > $L/ALL_DONE
