#!/bin/bash
# F1 paired run: one shard of one dataset. Usage: sbatch --export=DS=sugarcrepe,SHARD=0,NSHARDS=2 scripts/slurm/f1_paired.sbatch
# Submit from the project root; add your cluster's options on the command line (sbatch -p <partition> -A <account> ...).
# Set PY to the python of the cs707 environment if it is not the one on PATH.
#SBATCH --job-name=cs707_f1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/f1/f1_%x_%j.log
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HUB_OFFLINE=1
PY=${PY:-python}
echo "host $(hostname)"
for D in ${DS:-sugarcrepe textvqa}; do $PY -m exp.exp_f1_paired --dataset $D --shard ${SHARD:-0} --n_shards ${NSHARDS:-1} && $PY -m exp.exp_f1_paired --dataset $D --merge; done
