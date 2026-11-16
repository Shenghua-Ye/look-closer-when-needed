#!/bin/bash
# Download LLaVA-NeXT-7B at the pinned revision and verify the weight checksums (configs/model_sha256.txt).
set -e
PYTHON=${PYTHON:-python}
DIR=$($PYTHON -c "from huggingface_hub import snapshot_download as s; print(s('llava-hf/llava-v1.6-vicuna-7b-hf', revision='c916e6cdcd760b4cecd1dd4907f84ac649f93b23'))")
cd "$DIR" && sha256sum -c "$OLDPWD/configs/model_sha256.txt"
