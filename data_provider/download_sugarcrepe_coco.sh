#!/bin/sh
# Download SugarCrepe (7 JSON files) and COCO val2017 (images with boxes, from a Hugging Face mirror) into data/.
# Run from the project root (scripts/data_prepare.sh calls it). About 0.8 GB; files that exist are skipped.
#   detection-datasets/coco -> the 4,952 val2017 images that have annotations, with boxes
#   (the few SugarCrepe images without boxes are fetched from images.cocodataset.org by coco_export.py)
set -e
cd "$(dirname "$0")/../data"
HF=https://huggingface.co/datasets
mkdir -p hf sugarcrepe
for f in val-00000-of-00002-c4f2e391ee4aba11.parquet val-00001-of-00002-7af5414a3b178949.parquet; do
  [ -f hf/$f ] || curl -sSL -o hf/$f $HF/detection-datasets/coco/resolve/main/data/$f
done
for s in add_att add_obj replace_att replace_obj replace_rel swap_att swap_obj; do
  [ -f sugarcrepe/$s.json ] || curl -sSL -o sugarcrepe/$s.json \
    https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/data/$s.json
done
echo DONE
