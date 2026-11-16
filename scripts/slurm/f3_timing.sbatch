#!/bin/bash
# F3 online timing on round-0 TEST. Requests all 4 GPUs of one node, so that no other GPU job shares the node during timing.
# Submit from the project root; add your cluster's options on the command line (sbatch -p <partition> -A <account> ...).
# Set PY to the python of the cs707 environment if it is not the one on PATH.
#SBATCH --job-name=cs707_f3
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/f3/f3_%j.log
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
PY=${PY:-python}
echo "host $(hostname)"; nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
for D in ${DS:-sugarcrepe textvqa}; do $PY -m exp.exp_f3_timing --dataset $D $EXTRA; done
