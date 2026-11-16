"""Frozen TextVQA (val) manifests, joining the canonical image-level split.

    python run.py --task_name manifest --dataset textvqa     # or python data_provider/manifest_textvqa.py

- Images come from data/textvqa/ (exported from lmms-lab/textvqa validation parquet).
- Split: u ~ U(0,1) from the stream default_rng([20261001, 2]) over image ids sorted; same thresholds as SugarCrepe
  (DEV u < 0.4, CAL 0.4-0.7, TEST >= 0.7; pilot_dev = u < 0.3). A separate stream keeps SugarCrepe's split unchanged.
- Question types from rules on the question text only (rules v1, first match wins): reading, counting, existence, attribute, relation, other. Dataset identity is never a type label.
- Cross-source duplicate check against SugarCrepe images by MD5.
"""
import collections
import hashlib
import json
import re

import numpy as np

from root import DATA_DIR, MANIFEST_DIR, ROOT_DIR

ROOT = ROOT_DIR
SRC = DATA_DIR / "textvqa"
OUT = MANIFEST_DIR
RULES_VERSION = "qtype-v1"
RULES = [
    ("reading", [r"\b(read|reads|written|text|word|words|letter|letters)\b", r"\bwhat does\b.*\bsay\b", r"\bwhat number\b.*\bon\b"]),
    ("counting", [r"\bhow many\b", r"\bnumber of\b"]),
    ("existence", [r"\bis there\b", r"\bare there\b", r"\bcan you see\b"]),
    ("attribute", [r"\b(what|which)\b.*\b(color|colour|material|shape)\b"]),
    ("relation", [r"\bwhere\b", r"\bleft of\b", r"\bright of\b", r"\babove\b", r"\bbelow\b", r"\bbehind\b",
                  r"\bin front of\b", r"\bnext to\b"]),
]


def qtype(question):
    q = " ".join(question.lower().split())
    for name, pats in RULES:
        if any(re.search(p, q) for p in pats):
            return name
    return "other"


def build_manifest():
    qs = [json.loads(l) for l in open(SRC / "questions_val.jsonl")]
    ids = sorted({q["image_id"] for q in qs})
    u = dict(zip(ids, np.random.default_rng([20261001, 2]).random(len(ids))))
    sc_md5 = {x["md5"] for x in map(json.loads, open(OUT / "sugarcrepe_images.jsonl"))}
    images, dup = [], []
    meta_by_img = {q["image_id"]: q for q in qs}
    for i in ids:
        p = SRC / "images" / f"{i}.jpg"
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in sc_md5:
            dup.append(i)
        split = "DEV" if u[i] < 0.4 else ("CAL" if u[i] < 0.7 else "TEST")
        images.append(dict(image_id=i, path=str(p.relative_to(ROOT)), md5=h, width=meta_by_img[i]["width"],
                           height=meta_by_img[i]["height"], u=round(float(u[i]), 6), split=split, pilot_dev=bool(u[i] < 0.3)))
    split_of = {x["image_id"]: x for x in images}
    questions = [dict(qid=f"tv-{q['question_id']}", source="textvqa", image_id=q["image_id"], question=q["question"],
                      answers=q["answers"], qtype=qtype(q["question"]), split=split_of[q["image_id"]]["split"],
                      pilot_dev=split_of[q["image_id"]]["pilot_dev"]) for q in qs]
    with open(OUT / "textvqa_images.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in images)
    with open(OUT / "textvqa_questions.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in questions)
    cnt = lambda items, key: dict(sorted(collections.Counter(key(x) for x in items).items()))
    meta = dict(rules_version=RULES_VERSION, rules={k: v for k, v in RULES}, split_stream="default_rng([20261001, 2])",
                n_questions=len(questions), n_images=len(images), cross_source_duplicates=dup,
                images_per_split=cnt(images, lambda x: x["split"]), questions_per_split=cnt(questions, lambda x: x["split"]),
                pilot_dev_images=sum(x["pilot_dev"] for x in images), pilot_dev_questions=sum(x["pilot_dev"] for x in questions),
                qtype_all=cnt(questions, lambda x: x["qtype"]),
                qtype_images_per_split={t: cnt({(q["image_id"], q["split"]) for q in questions if q["qtype"] == t}, lambda z: z[1])
                                        for t in ["reading", "counting", "existence", "attribute", "relation", "other"]})
    (OUT / "textvqa_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "rules"}, indent=1))


if __name__ == "__main__":
    build_manifest()
