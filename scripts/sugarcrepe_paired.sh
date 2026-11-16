#!/bin/bash
# E2: paired cheap (base view) / full (AnyRes) A/B scoring on SugarCrepe DEV (~12 min for 1,000 questions).
# Output: results/E2_sugarcrepe_paired_dev/   Log: logs/paired/
# The plan is frozen once; paired runs resume, so re-running only adds missing questions.
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"
PYTHON=${PYTHON:-python}
LIMIT=0   # 0 = the whole plan
EXP_DIR=${EXP_DIR:-}   # new configuration -> new folder: EXP_DIR=E2_<name> bash scripts/sugarcrepe_paired.sh

if [ ! -f results/${EXP_DIR:-E2_sugarcrepe_paired_dev}/plan.jsonl ]; then   # 1,000-question plan, then the rest of pilot_dev (2,322 in all)
  $PYTHON run.py --task_name plan --dataset sugarcrepe ${EXP_DIR:+--exp_dir $EXP_DIR} --use_gpu False --is_logging False
  $PYTHON run.py --task_name extend_plan --dataset sugarcrepe ${EXP_DIR:+--exp_dir $EXP_DIR} --use_gpu False --is_logging False
fi
$PYTHON run.py --task_name paired --dataset sugarcrepe ${EXP_DIR:+--exp_dir $EXP_DIR} --limit ${LIMIT} --use_gpu True --is_logging True
