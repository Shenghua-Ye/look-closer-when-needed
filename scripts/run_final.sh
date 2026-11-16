#!/bin/bash
# Final study (CAL / TEST), in order. Each GPU step runs on one GPU at batch 1; set GPU=<index> (default 0).
# Steps that need the GPU are marked [GPU]; the others run on CPU in minutes.
#
#   bash scripts/run_final.sh check      # [GPU] F1 code reproduces 30 pilot_dev traces per dataset (DEV only)
#   bash scripts/run_final.sh f1         # [GPU] F1 paired runs on every non-DEV question, then merge (~2 GPU-h)
#   bash scripts/run_final.sh certify    # CPU: F2 step 1, certificates on CAL (all rounds, alphas, rules)
#   bash scripts/run_final.sh evaluate   # CPU: F2 step 2, TEST metrics (refuses to run before 'certify')
#   bash scripts/run_final.sh timing     # [GPU, idle node] F3 online latency on round-0 TEST, three arms, R = 3 (~4 GPU-h)
#   bash scripts/run_final.sh reducers   # [GPU] F4 pooling and VisionZip on round-0 TEST, then their evaluation
#   bash scripts/run_final.sh all        # everything above, in order
# On a Slurm cluster the same steps are submitted with scripts/slurm/*.sbatch (see scripts/README.md).
set -eo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONUNBUFFERED=1 PYTHONPATH="$PWD:$PYTHONPATH"
PYTHON=${PYTHON:-python}
mkdir -p logs/final
log() { tee -a "logs/final/$1_$(date +%Y%m%d_%H%M%S).log"; }

step() {
  case "$1" in
    check)    for DS in sugarcrepe textvqa; do $PYTHON -m exp.exp_f1_paired --dataset $DS --check_dev 30 2>&1 | log check_$DS; done ;;
    f1)       for DS in sugarcrepe textvqa; do $PYTHON -m exp.exp_f1_paired --dataset $DS 2>&1 | log f1_$DS
                                              $PYTHON -m exp.exp_f1_paired --dataset $DS --merge 2>&1 | log f1_$DS; done ;;
    certify)  $PYTHON -m exp.exp_f2_certify --certify 2>&1 | log certify ;;
    evaluate) $PYTHON -m exp.exp_f2_certify --evaluate 2>&1 | log evaluate ;;
    timing)   for DS in sugarcrepe textvqa; do $PYTHON -m exp.exp_f3_timing --dataset $DS 2>&1 | log timing_$DS; done ;;
    reducers) for DS in sugarcrepe textvqa; do for R in pool visionzip; do
                $PYTHON -m exp.exp_f4_reducers --dataset $DS --reducer $R 2>&1 | log reducers_${DS}_$R; done; done
              $PYTHON -m exp.exp_f4_reducers --evaluate 2>&1 | log reducers_eval ;;
    all)      for s in check f1 certify evaluate timing reducers; do step $s; done ;;
    *)        echo "unknown step: $1"; exit 1 ;;
  esac
}
step "${1:-all}"
