#!/bin/bash
# F2: certify on CAL (written once), then evaluate on TEST. CPU only; run after F1 has merged both datasets.
# Submit from the project root; add your cluster's options on the command line (sbatch -p <partition> -A <account> ...).
# Set PY to the python of the cs707 environment if it is not the one on PATH.
#SBATCH --job-name=cs707_f2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/f2/f2_%j.log
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD
PY=${PY:-python}
$PY -m exp.exp_f2_certify --certify && $PY -m exp.exp_f2_certify --evaluate
