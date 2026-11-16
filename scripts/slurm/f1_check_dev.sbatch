#!/bin/bash
#SBATCH --job-name=cs707_f1chk
# Submit from the project root; add your cluster's options on the command line (sbatch -p <partition> -A <account> ...).
# Set PY to the python of the cs707 environment if it is not the one on PATH.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/f1/check_dev_%j.log
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HUB_OFFLINE=1
PY=${PY:-python}
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
$PY -m exp.exp_f1_paired --dataset sugarcrepe --check_dev 30
$PY -m exp.exp_f1_paired --dataset textvqa --check_dev 30
