"""Build the frozen SugarCrepe manifests for the token-budget study.

    python run.py --task_name manifest --dataset sugarcrepe     # or python data_provider/manifest_sugarcrepe.py

Writes data/manifests/sugarcrepe_images.jsonl, sugarcrepe_questions.jsonl and manifest_meta.json.

- Question order follows the SugarCrepe files; which caption is option A is drawn once with seed 0.
- Splits are by image. Each image gets u ~ U(0,1) from seed 20261001 (images sorted by file name):
  DEV u < 0.4, CAL 0.4 <= u < 0.7, TEST u >= 0.7 (40/30/30, fixed). Development work uses only
  pilot_dev images (u < 0.3), a fixed subset of DEV; CAL and TEST stay untouched until the formal certification.
- obs_group is the deployable routing group: an A/B-symmetric edit signature of the two captions (rules below).
  The original SugarCrepe category is kept only as a diagnostic label.
"""
import collections
import hashlib
import json
import re

import numpy as np
from PIL import Image

from root import DATA_DIR, MANIFEST_DIR, ROOT_DIR

ROOT = ROOT_DIR
DATA = DATA_DIR
OUT = MANIFEST_DIR
CATEGORIES = ["replace_obj", "replace_att", "replace_rel", "swap_obj", "swap_att", "add_obj", "add_att"]  # demo order
SPLIT_SEED, AB_SEED = 20261001, 0
TOKEN_RE = r"[a-z0-9]+(?:'[a-z]+)?"  # applied to lowercased captions
RULES_VERSION = "edit-signature-v1"


def tokens(s):
    return re.findall(TOKEN_RE, s.lower())


def obs_group(a, b):
    """A/B-symmetric edit signature. Needs neither the correct caption nor the original category."""
    ta, tb = tokens(a), tokens(b)
    if ta == tb:
        return "other"
    ca, cb = collections.Counter(ta), collections.Counter(tb)
    if ca == cb:
        return "reorder"
    if (not (ca - cb) and len(tb) > len(ta)) or (not (cb - ca) and len(ta) > len(tb)):
        return "add_remove"
    if len(ta) == len(tb):
        return "replace"
    return "other"


def split_of(u):
    return "DEV" if u < 0.4 else ("CAL" if u < 0.7 else "TEST")


def build_manifest():
    OUT.mkdir(exist_ok=True)
    rows, src_hash = [], {}
    for cat in CATEGORIES:
        path = DATA / "sugarcrepe" / f"{cat}.json"
        raw = path.read_bytes()
        src_hash[cat] = hashlib.md5(raw).hexdigest()
        for key, ex in json.loads(raw).items():
            rows.append(dict(cat=cat, key=key, filename=ex["filename"], pos=ex["caption"], neg=ex["negative_caption"]))
    pos_first = np.random.default_rng(AB_SEED).random(len(rows)) < 0.5

    files = sorted({r["filename"] for r in rows})
    u = dict(zip(files, np.random.default_rng(SPLIT_SEED).random(len(files))))
    images, md5_seen = [], collections.defaultdict(list)
    for f in files:
        p = DATA / "val2017" / f
        h = hashlib.md5(p.read_bytes()).hexdigest()
        md5_seen[h].append(f)
        with Image.open(p) as im:
            w, hgt = im.size
        images.append(dict(image_id=f, path=str(p.relative_to(ROOT)), md5=h, width=w, height=hgt,
                           u=round(float(u[f]), 6), split=split_of(u[f]), pilot_dev=bool(u[f] < 0.3)))
    dup = {h: fs for h, fs in md5_seen.items() if len(fs) > 1}
    img_by_id = {im["image_id"]: im for im in images}

    questions = []
    for i, (r, pf) in enumerate(zip(rows, pos_first)):
        im = img_by_id[r["filename"]]
        a, b = (r["pos"], r["neg"]) if pf else (r["neg"], r["pos"])
        questions.append(dict(qid=f"sc-{r['cat']}-{r['key']}", row=i, source="sugarcrepe", image_id=r["filename"],
                              category=r["cat"], obs_group=obs_group(a, b), option_a=a, option_b=b,
                              correct="A" if pf else "B", split=im["split"], pilot_dev=im["pilot_dev"]))

    with open(OUT / "sugarcrepe_images.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in images)
    with open(OUT / "sugarcrepe_questions.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in questions)

    def count(items, key):
        return dict(sorted(collections.Counter(key(x) for x in items).items()))
    meta = dict(
        rules_version=RULES_VERSION, token_regex=TOKEN_RE, split_seed=SPLIT_SEED, ab_seed=AB_SEED,
        split_rule="DEV u<0.4, CAL 0.4<=u<0.7, TEST u>=0.7; pilot_dev = u<0.3", source_md5=src_hash,
        duplicate_images=dup,
        n_questions=len(questions), n_images=len(images),
        images_per_split=count(images, lambda x: x["split"]),
        pilot_dev_images=sum(x["pilot_dev"] for x in images),
        questions_per_split=count(questions, lambda x: x["split"]),
        pilot_dev_questions=sum(x["pilot_dev"] for x in questions),
        group_images_per_split={g: count({(q["image_id"], q["split"]) for q in questions if q["obs_group"] == g},
                                         lambda t: t[1]) for g in ["replace", "add_remove", "reorder", "other"]},
        category_by_group=count(questions, lambda x: f"{x['category']}|{x['obs_group']}"),
    )
    (OUT / "manifest_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ["n_questions", "n_images",
                                            "images_per_split", "pilot_dev_images", "questions_per_split",
                                            "pilot_dev_questions", "group_images_per_split"]}, indent=1))
    print("duplicate image files:", len(dup))


if __name__ == "__main__":
    build_manifest()
