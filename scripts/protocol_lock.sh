#!/bin/bash
# F0: lock the protocol before the formal CAL / TEST runs (CPU, under a minute; no model runs, no CAL/TEST outputs read).
# Candidate threshold paths from the DEV paired runs, the five CAL/TEST rounds (DEV fixed), the answer-blind CAL samples,
# eligibility from counts; then the supplement with the answer-blind TEST sample.
# Both are written once; a rerun that would change them stops.
# Output: results/F0_protocol_lock/   Log: logs/protocol_lock/
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PYTHON=${PYTHON:-python}

$PYTHON run.py --task_name protocol_lock --use_gpu False --is_logging True
$PYTHON run.py --task_name protocol_test_sample --use_gpu False --is_logging True
