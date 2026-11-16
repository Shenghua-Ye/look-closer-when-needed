"""F0 supplement: fix the TEST sample before any CAL or TEST model output is read. No model runs.

    python run.py --task_name protocol_test_sample --use_gpu False      # adds to results/F0_protocol_lock/

The F0 lock (protocol_lock.json, written by exp_protocol.py) holds the CAL samples but not the TEST sample. This task
leaves the lock untouched and writes a supplement next to it:
  - test_sample_<dataset>.json: per round, one question per (image, group) on that round's TEST images (the unit of
    the per-group TEST risk) and one per image (overall rate), drawn answer-blind with default_rng([TEST_SEED,
    dataset, round, 0 per group | 1 per image]), dataset sugarcrepe = 1, textvqa = 2
  - protocol_lock_test.json: the seed and the counts
The TEST images come from splits_<dataset>.json. Written once; a rerun that would change the supplement stops.
"""
import collections
import json

import numpy as np

from data_provider import datasets
from exp.exp_basic import ExpBasic
from exp.exp_protocol import CODE, OUT_DIR
from root import results_dir
from utils.tools import read_jsonl

TEST_SEED = 20261103


def test_sample(name, test_imgs, r):
    """Answer-blind TEST sample: one question per (image, group) and one per image (sorted keys, seeded draws)."""
    gk, test = datasets[name]["group_key"], set(test_imgs)
    by_g, by_img = collections.defaultdict(list), collections.defaultdict(list)
    for q in read_jsonl(datasets[name]["questions"]):
        if q["image_id"] in test:
            by_g[(q["image_id"], q[gk])].append(q["qid"])
            by_img[q["image_id"]].append(q["qid"])
    rng_g = np.random.default_rng([TEST_SEED, CODE[name], r, 0])
    rng_i = np.random.default_rng([TEST_SEED, CODE[name], r, 1])
    per_group = [dict(image_id=i, group=g, qid=sorted(v)[rng_g.integers(len(v))]) for (i, g), v in sorted(by_g.items())]
    per_image = [dict(image_id=i, qid=sorted(v)[rng_i.integers(len(v))]) for i, v in sorted(by_img.items())]
    return per_group, per_image


def build_supplement(lock_dir):
    path = lock_dir / "protocol_lock.json"
    lock = json.loads(path.read_text())
    files, sup = {}, dict(
        seed=TEST_SEED,
        stream="default_rng([seed, dataset, round, 0 per (image, group) | 1 per image]); dataset sugarcrepe = 1, textvqa = 2",
        unit="per-group TEST risk: the mean loss over the sampled question of each (image, group) of the group, i.e. "
             "image-weighted; overall rate: one question per image. All TEST questions, image-weighted, stay a diagnostic.",
        datasets={})
    for name in datasets:
        sp_path = lock_dir / f"splits_{name}.json"
        splits = json.loads(sp_path.read_text())
        groups = lock["groups"][name]
        samples, rec = [], []
        for r, sp in enumerate(splits["rounds"]):
            assert not set(sp["TEST"]) & set(sp["CAL"]) and not set(sp["TEST"]) & set(splits["DEV"])
            per_group, per_image = test_sample(name, sp["TEST"], r)
            samples.append(dict(round=r, per_group=per_group, per_image=per_image))
            n = collections.Counter(x["group"] for x in per_group)
            rec.append(dict(round=r, n_test_images=len(sp["TEST"]), n_per_group=len(per_group), n_per_image=len(per_image),
                            n_by_group={g: n[g] for g in groups}))
        files[f"test_sample_{name}.json"] = json.dumps(samples)
        sup["datasets"][name] = dict(rounds=rec)
    return sup, files


def write_report(lock_dir, sup):
    L = ["# F0 supplement: TEST sample", "",
         "Written before any CAL or TEST model output was read. The candidate thresholds and the lock itself are "
         "unchanged.", "",
         f"- Seed {sup['seed']}: {sup['stream']}.", f"- Unit: {sup['unit']}", "",
         "| Dataset | Round | TEST images | samples per (image, group) | per group |", "|---|---|---|---|---|"]
    for name, d in sup["datasets"].items():
        for r in d["rounds"]:
            L.append(f"| {name} | {r['round']} | {r['n_test_images']} | {r['n_per_group']} | " +
                     ", ".join(f"{g} {k}" for g, k in r["n_by_group"].items()) + " |")
    (lock_dir / "protocol_lock_test.md").write_text("\n".join(L) + "\n")


def run_test_sample(exp_dir=""):
    lock_dir = results_dir(exp_dir, OUT_DIR)
    sup, files = build_supplement(lock_dir)
    sup = json.loads(json.dumps(sup))
    path = lock_dir / "protocol_lock_test.json"
    if path.exists():
        if json.loads(path.read_text()) != sup:
            raise SystemExit(f"{path} holds a different TEST sample. It is written once; delete it and the "
                             "test_sample_*.json files to rebuild.")
        print(f"{path} unchanged")
    else:
        for f, text in files.items():
            (lock_dir / f).write_text(text)
        path.write_text(json.dumps(sup, indent=1))
    write_report(lock_dir, sup)
    print(f"TEST sample: {lock_dir}")


class ExpProtocolTest(ExpBasic):
    def run(self):
        run_test_sample(self.args.exp_dir)
