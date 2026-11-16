#!/bin/bash
# E8: confidence signals of the TextVQA cheap answers on DEV (GPU ~5 min, no timing).
# Output: results/E8_textvqa_signals_dev/   Log: logs/signals/
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"
PYTHON=${PYTHON:-python}

$PYTHON run.py --task_name signals --dataset textvqa --use_gpu True --is_logging True
