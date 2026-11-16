#!/bin/bash
# F4 reducers on round-0 TEST. Usage: sbatch --export=ALL,DS=sugarcrepe,RED=pool scripts/slurm/f4_reducers.sbatch
# Submit from the project root; add your cluster's options on the command line (sbatch -p <partition> -A <account> ...).
# Set PY to the python of the cs707 environment if it is not the one on PATH.
#SBATCH --job-name=cs707_f4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/f4/f4_%x_%j.log
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HUB_OFFLINE=1
PY=${PY:-python}
echo "host $(hostname)"
for D in sugarcrepe textvqa; do for R in pool visionzip; do $PY -m exp.exp_f4_reducers --dataset $D --reducer $R; done; done
