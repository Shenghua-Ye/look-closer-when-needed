# Data Provider

Everything under `data/` is an **input**. Code that prepares it lives here; nothing in this folder writes to `results/`.
Run all of it with `bash scripts/data_prepare.sh` (or `bash scripts/run.sh prepare`) from the project root.

## Layout of `data/`

| Path | Content | Produced by |
|------|---------|-------------|
| `data/sugarcrepe/` | 7 SugarCrepe JSON files (7,511 questions) | `download_sugarcrepe_coco.sh` |
| `data/hf/val-*.parquet` | COCO val2017 images with boxes (Hugging Face mirror) | `download_sugarcrepe_coco.sh` |
| `data/val2017/`, `data/annotations/instances_val2017.json` | COCO val2017 images (SugarCrepe and POPE) and boxes (for the POPE size types) | `coco_export.py` |
| `data/hf/textvqa/`, `data/hf/pope/` | TextVQA validation and POPE test parquet files | `_downloading.py` |
| `data/textvqa/` | TextVQA validation exported to `images/*.jpg` and `questions_val.jsonl` | `textvqa_export.py` |
| `data/manifests/`, `data/manifests_pope/` | **Frozen** question and image manifests with the splits (included in the repository) | `manifest_*.py` |

`_downloading.py` also downloads LLaVA-NeXT-7B at the pinned revision into the Hugging Face cache;
`scripts/download_model.sh` then checks the weights against `configs/model_sha256.txt`.

## Manifests and splits

| File | Rows | Notes |
|------|------|-------|
| `sugarcrepe_images.jsonl` | 1,560 images | split by `u` from seed 20261001 (images sorted by file name) |
| `sugarcrepe_questions.jsonl` | 7,511 questions | which caption is option A is drawn with seed 0; `obs_group` = caption-edit type; `category` = the original SugarCrepe label, for diagnostics only |
| `textvqa_images.jsonl` | 3,166 images | split by `u` from the stream `default_rng([20261001, 2])` |
| `textvqa_questions.jsonl` | 5,000 questions | `qtype` from the question-text rules |
| `manifests_pope/` | 5,127 questions on 500 images | size-tercile types; see `manifest_pope.py` |

- Split rule: `DEV u < 0.4`, `CAL 0.4 ≤ u < 0.7`, `TEST u ≥ 0.7` (40/30/30 by image, the proportions of SAGE-CP).
  The DEV paired runs use `pilot_dev` (`u < 0.3`), a fixed subset of DEV.
- Rebuilding the manifests with the same code reproduces the files byte for byte. They must not be rebuilt with other
  seeds or rules once calibration data have been used.
- `obs_group` (SugarCrepe): an edit signature of the two captions that does not change when they are swapped —
  `reorder`, `add_remove`, `replace`, `other`.
- `qtype` (TextVQA): the first matching rule — `reading`, `counting`, `existence`, `attribute`, `relation`, else
  `other` (Appendix D of the report lists the keywords).

Dataset registry: `data_provider/__init__.py`.
