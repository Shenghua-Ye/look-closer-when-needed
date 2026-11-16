#!/bin/bash
# Prepare all inputs under data/. Run from the project root: bash scripts/data_prepare.sh (or bash scripts/run.sh prepare)
# Estimated: ~15 GB download (model 13 GB). Files that exist are skipped.
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
PYTHON=${PYTHON:-python}              # the cs707 environment (environment.yml)
PYTHON_ARROW=${PYTHON_ARROW:-$PYTHON} # needs pyarrow (included in environment.yml)

sh data_provider/download_sugarcrepe_coco.sh        # SugarCrepe JSON, COCO val2017 parquet
$PYTHON_ARROW data_provider/coco_export.py           # data/val2017/*.jpg + data/annotations/instances_val2017.json
$PYTHON data_provider/_downloading.py                # LLaVA-NeXT-7B (pinned revision), TextVQA and POPE parquet
PYTHON=$PYTHON bash scripts/download_model.sh       # verify the model weights (configs/model_sha256.txt)
$PYTHON_ARROW data_provider/textvqa_export.py        # data/textvqa/images + questions_val.jsonl
$PYTHON run.py --task_name manifest --dataset sugarcrepe
$PYTHON run.py --task_name manifest --dataset textvqa
$PYTHON_ARROW data_provider/manifest_pope.py         # POPE manifests (data/manifests_pope/) and images
