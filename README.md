# Look Closer When Needed: Risk-Controlled Visual Budgets for Vision-Language Models

Code, data preparation and results for the CS707 final report.

A frozen LLaVA-NeXT-7B first answers from one low-resolution view of the image (577 visual tokens). If it is
confident enough for the question's type, that answer is returned; otherwise it continues at full resolution
(up to 2,928 tokens), reusing the computation it has already done. One confidence threshold per question type is
chosen on a calibration split with Learn-then-Test, so that with probability at least 1 − δ the returned answer
differs from the model's own full-resolution answer on at most a fraction α of requests in every type. Datasets:
SugarCrepe, TextVQA and POPE.

## Setup

```bash
conda env create -f environment.yml && conda activate cs707     # Python 3.10, PyTorch 2.5.1 (CUDA 12.4), Transformers 4.47.0
bash scripts/run.sh prepare                                     # data and model (~15 GB), weight checksums, manifests
```

Hardware: one NVIDIA GPU with at least 24 GB of memory per process (peak use 14.9 GiB in half precision), batch
size 1. The timing step should run on a node with no other GPU work. `requirements.txt` lists the same pinned
packages for pip.

## Reproduce

```bash
bash scripts/run.sh dev       # DEV paired runs, TextVQA confidence signals, then splits, samples and candidate thresholds
bash scripts/run.sh main      # SugarCrepe + TextVQA, in order: check, paired runs, thresholds, TEST metrics, timing, reducers
bash scripts/run.sh pope      # POPE, in order: DEV runs, DEV report, rehearsal, lock, CAL/TEST runs, thresholds, TEST metrics, reducers, timing
bash scripts/run.sh redraws   # all 1,000 calibration/test rounds, replayed from the paired runs (CPU), plus VisionZip on round-0 CAL
bash scripts/run.sh assets    # every data table and figure of the report (paper_results.ipynb)
```

`bash scripts/run.sh all` runs `prepare` and all of the above in order.

Each step can also be run on its own (`bash scripts/run_final.sh <step>`; see `scripts/README.md`). On a Slurm
cluster, `scripts/slurm/*.sbatch` submit the GPU steps; add your partition and account on the `sbatch` command line.

Thresholds are always chosen from CAL alone and written before any TEST metric is computed; the evaluation steps
refuse to run otherwise. [`paper_results.ipynb`](paper_results.ipynb) turns the outputs into the tables and figures of
the report in a few seconds on a CPU; it is saved with the outputs of our run, so it can be read on GitHub without
running it.

## Outputs

Every step writes to `results/`:

| Folder | Contents |
|---|---|
| `results/E2_sugarcrepe_paired_dev/`, `results/E5_textvqa_paired_dev/`, `results/E8_textvqa_signals_dev/` | Cheap and full pass on the DEV questions (and the TextVQA confidence signals); the candidate thresholds are built from these |
| `results/F0_protocol_lock/` | Splits (five rounds), CAL and TEST samples, candidate thresholds per type, fixed before CAL |
| `results/F1_paired_caltest/<dataset>/` | Cheap pass, full pass and forced fallback on every CAL and TEST question, with stage times |
| `results/F2_certify_evaluate/` | Thresholds from CAL (`certificates.json`, `cal_report.md`) and TEST results (`test_report.md`, `test_results.json`) |
| `results/F3_timing_test/<dataset>/` | Measured latency on round-0 TEST: full, restart and resume |
| `results/F4_reducers_test/` | 2×2 pooling and VisionZip on round-0 TEST |
| `results/F6_rounds_extra/` | All 1,000 CAL/TEST rounds at three values of α (`rounds.json.gz`, summaries, report) |
| `results/F7_reducers_cal/` | VisionZip on every round-0 CAL question, so that the fixed cuts can be counted over the same 1,000 rounds |
| `results/E9_pope_paired_dev/`, `results/P0_pope_protocol_lock/` | POPE: DEV runs and DEV report, fixed protocol |
| `results/P1_pope_paired_caltest/` … `results/P4_pope_reducers_test/` | POPE: CAL/TEST runs, thresholds and TEST results, measured latency, pooling and VisionZip |


## Code layout

| Path | Role |
|---|---|
| `models/llava_next.py` | Cheap pass (base view), full pass (native AnyRes), forced-choice scoring and decoding |
| `exp/exp_reuse.py` | The fallback that continues from the cached prefix instead of starting over |
| `exp/exp_frontier.py` | DEV data, cost model and candidate thresholds |
| `exp/exp_protocol.py`, `exp/exp_protocol_test.py` | The fixed splits, samples and candidate thresholds (F0) |
| `exp/exp_f1_paired.py` … `exp/exp_f4_reducers.py` | Main study: paired runs, thresholds and TEST metrics, timing, reducers |
| `exp/exp_pope_*.py`, `exp/pope_common.py`, `data_provider/manifest_pope.py` | POPE |
| `exp/exp_rounds_extra.py`, `exp/exp_reducers_cal.py` | The 1,000 CAL/TEST rounds; VisionZip on round-0 CAL |
| `exp/exp_paired_*.py`, `exp/exp_signals.py` | The DEV paired runs and the TextVQA confidence signals |
| `exp/exp_pooled.py`, `exp/timing_tools.py`, `exp/sugarcrepe_inputs.py` | Token reducers; cache and GPU-exclusivity helpers; SugarCrepe inputs |
| `exp/exp_human_eval.py` | Instruments of the two-annotator check of changed TextVQA answers (Appendix E) |
| `data_provider/` | Downloads and frozen manifests (splits by image, 40/30/30) |
| `configs/final.yaml` | Documentation of the constants the code uses (the code is the source of truth) |
| `paper_results.ipynb` | Every data table and figure of the report |

`exp/exp_pooled.py` ports VisionZip from its official implementation (Apache-2.0). Datasets keep their own
licenses (data table in Appendix D of the report).

## Dataset images and model weights

`bash scripts/run.sh prepare` downloads them from:

| Data | Source |
|---|---|
| SugarCrepe | [RAIVNLab/sugar-crepe](https://github.com/RAIVNLab/sugar-crepe) (`data/*.json`) |
| COCO val2017 images | [detection-datasets/coco](https://huggingface.co/datasets/detection-datasets/coco) (validation split); images without boxes from [images.cocodataset.org](http://images.cocodataset.org/val2017/) |
| TextVQA (validation) | [lmms-lab/textvqa](https://huggingface.co/datasets/lmms-lab/textvqa) |
| POPE | [lmms-lab/POPE](https://huggingface.co/datasets/lmms-lab/POPE) |
| LLaVA-NeXT-7B | [llava-hf/llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf), revision `c916e6cd`; weights checked against `configs/model_sha256.txt` |

## License

Our code is released under the MIT License (`LICENSE`). `exp/exp_pooled.py` ports parts of VisionZip, which is
Apache-2.0. The datasets and the model keep their own licenses.
