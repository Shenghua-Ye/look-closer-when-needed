"""Download what the experiments need: LLaVA-NeXT-7B at the pinned revision (into the Hugging Face cache) and the
evaluation splits of TextVQA and POPE (parquet with embedded images) into data/hf/. SugarCrepe and COCO val2017 come
from data_provider/download_sugarcrepe_coco.sh. scripts/download_model.sh then checks the model weights against
configs/model_sha256.txt.

    python data_provider/_downloading.py
"""
from huggingface_hub import snapshot_download

from root import DATA_DIR

HF_DIR = DATA_DIR / "hf"
MODEL = "llava-hf/llava-v1.6-vicuna-7b-hf"
REVISION = "c916e6cdcd760b4cecd1dd4907f84ac649f93b23"  # the revision every run used (models/llava_next.py)
JOBS = [
    ("lmms-lab/textvqa", ["data/validation-*.parquet"], "textvqa"),
    ("lmms-lab/POPE", ["data/test-*.parquet"], "pope"),
]


def download():
    for repo, patterns, name in JOBS:
        out = snapshot_download(repo, repo_type="dataset", allow_patterns=patterns, local_dir=HF_DIR / name)
        print("dataset ready:", out, flush=True)
    path = snapshot_download(MODEL, revision=REVISION)
    print("model ready:", path, flush=True)


if __name__ == "__main__":
    download()
