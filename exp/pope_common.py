"""Shared constants and helpers for the POPE extension (exp_pope_*.py)."""
import json
from pathlib import Path

from root import RESULTS_DIR, ROOT_DIR

MAN = ROOT_DIR / "data" / "manifests_pope"
DEV_DIR = RESULTS_DIR / "E9_pope_paired_dev"
LOCK_DIR = RESULTS_DIR / "P0_pope_protocol_lock"
P1_DIR = RESULTS_DIR / "P1_pope_paired_caltest"
P2_DIR = RESULTS_DIR / "P2_pope_certify_evaluate"
GROUPS = ["small", "medium", "large"]
GROUP_KEY = "size_type"
CODE = 3                                   # dataset index in all POPE seed streams (sugarcrepe 1, textvqa 2)
SEEDS = dict(order=20261201, path_order=20261202, rounds=20261101, cal_sample=20261102, test_sample=20261103,
             rehearsal=20261203, bootstrap=20261106)
YES = ("▁Yes", "Yes", "▁yes", "yes")       # forced-choice token variants, fixed before any POPE model output
NO = ("▁No", "No", "▁no", "no")
ALPHA_MAIN = 0.05


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def questions():
    return read_jsonl(MAN / "pope_questions.jsonl")


def images():
    return {x["image_id"]: x for x in read_jsonl(MAN / "pope_images.jsonl")}


def load_lock():
    p = LOCK_DIR / "pope_lock.json"
    if not p.exists():
        raise SystemExit("no POPE lock: run exp_pope_protocol.py --lock first")
    return json.loads(p.read_text())
