# Scripts Guide

Run every script from the **project root** (not from `scripts/`):

```bash
bash scripts/<script_name>.sh
```

Each script exports `CUDA_VISIBLE_DEVICES`, `PYTHONUNBUFFERED` and `PYTHONPATH`, then calls `run.py` or a module in
`exp/`. Use `PYTHON=/path/to/env/bin/python bash scripts/...` if the conda environment is not activated. Results go to
`results/<experiment>/` (the Output column below); logs go to `logs/<task_name>/<setting>_<timestamp>.log`.

`bash scripts/run.sh` is the single entry point (`prepare`, `dev`, `main`, `pope`, `redraws`, `assets`, `all`); the sections
below list the individual steps.

## 1. Data preparation

| Script | What | Runtime |
|--------|------|---------|
| `data_prepare.sh` | Download SugarCrepe, COCO val2017, TextVQA, POPE and the model; check the weights against `configs/model_sha256.txt`; export the images; build `data/manifests/` and `data/manifests_pope/` | ~15 GB download |

Rebuilding the manifests reproduces them byte for byte.

## 2. DEV paired runs

Run these first (`bash scripts/run.sh dev`): the candidate thresholds are built from them.

| Script | What | Output | Runtime (one GPU) |
|--------|------|--------|-------------------|
| `sugarcrepe_paired.sh` | Cheap and full pass on the SugarCrepe DEV plan | `results/E2_sugarcrepe_paired_dev/` | ~30 min |
| `textvqa_paired.sh` | Cheap and full pass on the TextVQA DEV plan (after a decode-loop check) | `results/E5_textvqa_paired_dev/` | ~25 min |
| `signals.sh` | Confidence signals of the TextVQA cheap answers | `results/E8_textvqa_signals_dev/` | ~5 min |

- **Plans are frozen.** The scripts build each plan once from fixed seeds and extend it to every pilot_dev question
  (2,322 SugarCrepe and 1,466 TextVQA questions); `plan` refuses to overwrite an existing `plan.jsonl`.
- **Paired runs resume.** Re-running adds only questions missing from `traces.jsonl`.
- **New configuration, new folder.** A paired run compares its folder's `run_config.json` with the current
  configuration and stops on any difference; run a new configuration in a sibling folder
  (`EXP_DIR=E5_textvqa_<name> bash scripts/textvqa_paired.sh`).
- **Timing needs a quiet GPU.** Stage times are part of the results, so do not share the card with other jobs.

## 3. Final study (CAL / TEST)

One command per step, in this order (`bash scripts/run_final.sh <step>`, or `all`; `bash scripts/run.sh main` runs
them all). On a Slurm cluster the GPU steps are the jobs in `scripts/slurm/`; submit them from the project root and add
your partition and account on the command line (`sbatch -p <partition> -A <account> scripts/slurm/<job>.sbatch`).

| Step | What | Command | Slurm | GPU time | Output |
|---|---|---|---|---|---|
| lock | Splits for five rounds, CAL and TEST samples, eligibility, candidate thresholds per type; before any CAL output | `bash scripts/protocol_lock.sh` | — | CPU, <1 min | `results/F0_protocol_lock/` |
| check | The paired-run code reproduces 30 DEV traces per dataset | `run_final.sh check` | `f1_check_dev.sbatch` | ~2 min | `results/F1_paired_caltest/<ds>/check_dev.json` |
| f1 | Paired runs (full, cheap with the cache kept, forced fallback) on all 7,585 non-DEV questions | `run_final.sh f1` | `f1_paired.sbatch` | ~1 h | `results/F1_paired_caltest/<ds>/traces.jsonl` |
| certify | Thresholds on CAL: 5 rounds × 3 α × 7 rules × configurations | `run_final.sh certify` | `f2_certify.sbatch` (CPU; also runs evaluate) | minutes | `results/F2_certify_evaluate/certificates.json` |
| evaluate | TEST metrics (refuses to run before certify) | `run_final.sh evaluate` | CPU (in `f2_certify.sbatch`) | minutes | `results/F2_certify_evaluate/test_report.md` |
| timing | Measured latency on round-0 TEST: full / restart / resume, 3 repeats, no other GPU job on the node | `run_final.sh timing` | `f3_timing.sbatch` | ~3–4 h | `results/F3_timing_test/<ds>/summary.md` |
| reducers | 2×2 pooling and VisionZip on round-0 TEST | `run_final.sh reducers` | `f4_reducers.sbatch` | ~1 h | `results/F4_reducers_test/summary.md` |
| redraws | All 1,000 CAL/TEST rounds, replayed from the paired runs; VisionZip on round-0 CAL | `bash scripts/run.sh redraws` | — | CPU ~15 min + GPU ~30 min | `results/F6_rounds_extra/`, `results/F7_reducers_cal/` |
| assets | Every data table and figure of the report | `bash scripts/run.sh assets` | — | CPU, seconds | `paper_results.ipynb` |
| human | Two-annotator check of changed TextVQA answers | `python -m exp.exp_human_eval --build` / `--analyze` | — | — | `report/human_eval/` |

The timing step checks with `nvidia-smi` that no other process uses a GPU before and during the run; `--force` runs
anyway and records it. `--dry_run` prints the request list, thresholds and replayed predictions without a model or GPU.

## 4. POPE (`bash scripts/run.sh pope`)

| Step | What | Command | Output |
|---|---|---|---|
| manifest | 5,127 unique (image, object) questions on 500 images; size-tercile types; 40/30/30 split by image | `python data_provider/manifest_pope.py` | `data/manifests_pope/` |
| regression | The POPE runner with the A/B scorer reproduces the SugarCrepe paired runs | `python -m exp.exp_pope_paired --regress_sc 30` | `results/E9_pope_paired_dev/regress_sugarcrepe.json` |
| dev | Paired runs on all DEV questions (+ `--merge`); `NGPU` shards in parallel | `python -m exp.exp_pope_paired --stage dev` | `results/E9_pope_paired_dev/traces.jsonl` |
| report | DEV report, then the whole pipeline rehearsed on part of DEV | `python -m exp.exp_pope_protocol --dev_report`; `python -m exp.exp_pope_certify --rehearse` | `dev_report.md`, `rehearsal_report.md` |
| lock | Rounds, CAL/TEST samples, eligibility, candidate thresholds; written once | `python -m exp.exp_pope_protocol --lock` | `results/P0_pope_protocol_lock/` |
| caltest | Paired runs on all CAL and TEST questions (+ `--merge`); refuses without the lock | `python -m exp.exp_pope_paired --stage caltest` | `results/P1_pope_paired_caltest/traces.jsonl` |
| certify, evaluate | Thresholds from CAL, then TEST metrics and per-setting slices | `python -m exp.exp_pope_certify --certify --evaluate` | `results/P2_pope_certify_evaluate/` |
| reducers | 2×2 pooling and VisionZip on round-0 TEST, then their evaluation | `python -m exp.exp_pope_reducers --reducer pool` (and `visionzip`), then `--evaluate` | `results/P4_pope_reducers_test/` |
| timing | Measured latency on round-0 TEST | `python -m exp.exp_pope_timing` | `results/P3_pope_timing_test/` |
