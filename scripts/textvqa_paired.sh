#!/bin/bash
# E5: paired cheap / full short-answer generation on TextVQA DEV (~15 min for 1,000 questions).
# Output: results/E5_textvqa_paired_dev/   Log: logs/paired/
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"
PYTHON=${PYTHON:-python}
LIMIT=0   # 0 = the whole plan
EXP_DIR=${EXP_DIR:-}   # new configuration -> new folder: EXP_DIR=E5_<name> bash scripts/textvqa_paired.sh
D=results/${EXP_DIR:-E5_textvqa_paired_dev}

if [ ! -f $D/plan.jsonl ]; then   # 1,000-question plan, then the rest of pilot_dev (1,466 in all)
  $PYTHON run.py --task_name plan --dataset textvqa ${EXP_DIR:+--exp_dir $EXP_DIR} --use_gpu False --is_logging False
  $PYTHON run.py --task_name extend_plan --dataset textvqa ${EXP_DIR:+--exp_dir $EXP_DIR} --use_gpu False --is_logging False
fi
[ -f $D/decode_check.json ] || $PYTHON run.py --task_name check_decode --dataset textvqa ${EXP_DIR:+--exp_dir $EXP_DIR} --use_gpu True
$PYTHON run.py --task_name paired --dataset textvqa ${EXP_DIR:+--exp_dir $EXP_DIR} --limit ${LIMIT} --use_gpu True --is_logging True
