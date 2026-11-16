#!/bin/bash
# One entry point for the artifact. Run from anywhere; it works in the project root.
#
#   bash scripts/run.sh prepare   # download data and model (~15 GB), verify the weights, build every manifest
#   bash scripts/run.sh dev       # DEV paired runs (SugarCrepe, TextVQA), TextVQA confidence signals, then the protocol
#                                 # lock: splits, samples and candidate thresholds fixed before CAL (~1 GPU-hour)
#   bash scripts/run.sh main      # SugarCrepe + TextVQA: paired runs, thresholds on CAL, TEST metrics, timing,
#                                 # token reducers (scripts/run_final.sh all; ~6 GPU-hours)
#   bash scripts/run.sh pope      # POPE: paired runs on DEV, DEV report, rehearsal, lock, paired runs on CAL/TEST,
#                                 # thresholds on CAL, TEST metrics, token reducers, online timing (~3 GPU-hours)
#   bash scripts/run.sh redraws   # all 1,000 CAL/TEST rounds at three alphas, replayed from the finished runs (CPU, ~15 min),
#                                 # plus VisionZip on round-0 CAL (GPU, ~30 min) for the two fixed cuts over the same rounds
#   bash scripts/run.sh assets    # re-run paper_results.ipynb: every table and figure of the report (CPU, seconds)
#   bash scripts/run.sh all       # prepare, dev, main, pope, redraws, assets
#
# Options (environment variables): PYTHON (default: python of the active cs707 environment), NGPU (GPUs used in
# parallel for the POPE paired runs, default 1), GPU (the GPU index for scripts/run_final.sh, default 0).
#
# Every step writes to results/ and never overwrites a lock or thresholds that exist.
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH" PYTHONUNBUFFERED=1
PYTHON=${PYTHON:-python}
NGPU=${NGPU:-1}

pope_paired() {  # $1 = dev | caltest; one shard per GPU, then merge
  mkdir -p logs/pope
  for ((k = 0; k < NGPU; k++)); do
    CUDA_VISIBLE_DEVICES=$k $PYTHON -m exp.exp_pope_paired --stage "$1" --shard $k --n_shards "$NGPU" > "logs/pope/$1_shard$k.log" 2>&1 &
  done
  wait
  $PYTHON -m exp.exp_pope_paired --stage "$1" --merge
}

case "${1:-all}" in
  prepare) PYTHON=$PYTHON bash scripts/data_prepare.sh ;;
  dev)     for s in sugarcrepe_paired textvqa_paired signals protocol_lock; do PYTHON=$PYTHON bash scripts/$s.sh; done ;;
  main)    PYTHON=$PYTHON bash scripts/run_final.sh all ;;
  pope)    pope_paired dev
           $PYTHON -m exp.exp_pope_protocol --dev_report
           $PYTHON -m exp.exp_pope_certify --rehearse
           $PYTHON -m exp.exp_pope_protocol --lock
           pope_paired caltest
           $PYTHON -m exp.exp_pope_certify --certify --evaluate
           for R in pool visionzip; do CUDA_VISIBLE_DEVICES=0 $PYTHON -m exp.exp_pope_reducers --reducer $R; done
           $PYTHON -m exp.exp_pope_reducers --evaluate
           CUDA_VISIBLE_DEVICES=0 $PYTHON -m exp.exp_pope_timing ;;   # needs a node with no other GPU work
  redraws) $PYTHON -m exp.exp_rounds_extra --rounds 1000
           for D in sugarcrepe textvqa pope; do CUDA_VISIBLE_DEVICES=0 $PYTHON -m exp.exp_reducers_cal --dataset $D; done   # GPU, VisionZip on round-0 CAL
           $PYTHON -m exp.exp_rounds_extra --fixed --rounds 1000 ;;
  assets)  $PYTHON -m jupyter nbconvert --to notebook --execute --inplace paper_results.ipynb ;;
  all)     bash "$0" prepare && bash "$0" dev && bash "$0" main && bash "$0" pope && bash "$0" redraws && bash "$0" assets ;;
  *)       echo "usage: bash scripts/run.sh [prepare|dev|main|pope|redraws|assets|all]"; exit 1 ;;
esac
