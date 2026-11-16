"""Frozen POPE manifests for the POPE extension (a third dataset, added after the main study).

    python data_provider/manifest_pope.py    # CPU only; needs pyarrow

Inputs: data/hf/pope/data/test-0000{0,1,2}-of-00003.parquet (lmms-lab POPE, 9,000 rows on 500 COCO val2014 images),
data/annotations/instances_val2017.json (COCO val2017 boxes, for the object-size table), the SugarCrepe image manifest
(for the overlap check only).

Outputs (a separate folder, so nothing of the main study's data/manifests/ changes):
  data/pope/images/<image_id>.jpg                      image bytes exactly as stored in the parquet
  data/manifests_pope/pope_images.jsonl                 one row per image: id, path, md5, size, u, split, pilot_dev
  data/manifests_pope/pope_questions.jsonl              one row per unique (image, object) pair
  data/manifests_pope/pope_meta.json                    rules, the 79-row size table, counts

Rules (none uses a model output):
  - Unit: one request per unique (image, queried object). The three POPE settings share most questions; 57 adversarial
    rows spell "imange"; duplicates collapse to one request. The canonical question text is the correctly spelled one
    when it exists. The settings and original row ids of every pair are kept for descriptive slices.
  - Type: size tercile of the queried object. For each of the 79 COCO categories, the median relative box area
    (bbox width x height / image area) over COCO val2017 annotations, excluding the val2017 images that are POPE images
    (so no POPE image contributes). Terciles over the 79 objects' medians: small < t1 <= medium < t2 <= large, with
    t1 = sorted[26], t2 = sorted[52] (0-based). Observable from the request text alone.
  - Split by image, 40/30/30 with the main study's thresholds; u from default_rng([20261001, 3]) over sorted image ids
    (dataset code 3). pilot_dev = u < 0.3 builds the DEV paths; 0.3 <= u < 0.4 is the DEV rehearsal band.
  - Images identical (MD5) to SugarCrepe images are kept and listed.
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "hf" / "pope" / "data"
IMG_DIR = ROOT / "data" / "pope" / "images"
OUT = ROOT / "data" / "manifests_pope"
ANN = ROOT / "data" / "annotations" / "instances_val2017.json"
SC_IMAGES = ROOT / "data" / "manifests" / "sugarcrepe_images.jsonl"
SPLIT_STREAM = [20261001, 3]
RULES_VERSION = "pope-size-tercile-v1"
SETTINGS = ("random", "popular", "adversarial")
QUESTION_RE = re.compile(r"^is there an? (.+?) in the ima?n?ge\?$")
# COCO's 80 categories in the standard order; the local instances_val2017.json uses contiguous ids 0..79 in this order
COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
          "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
          "hair drier", "toothbrush"]


def obj_of(question):
    m = QUESTION_RE.match(" ".join(question.lower().split()))
    if not m:
        raise ValueError(f"unexpected POPE question: {question!r}")
    return m.group(1)


def load_rows():
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(SRC.glob("test-*.parquet")):
        t = pq.read_table(f)
        for r in t.to_pylist():
            rows.append(r)
    return rows


def size_table(objects, pope_coco_ids):
    d = json.loads(ANN.read_text())
    assert "categories" not in d, "this loader assumes the trimmed file with contiguous ids"
    area = {im["id"]: im["width"] * im["height"] for im in d["images"]}
    counts = collections.Counter(a["category_id"] for a in d["annotations"])
    assert sorted(counts) == list(range(80)) and counts.most_common(1)[0][0] == 0, "category ids are not COCO80 order (person first)"
    rel = collections.defaultdict(list)
    for a in d["annotations"]:
        if a["image_id"] in pope_coco_ids:
            continue
        w, h = a["bbox"][2], a["bbox"][3]
        rel[COCO80[a["category_id"]]].append(w * h / area[a["image_id"]])
    med = {o: float(np.median(rel[o])) for o in objects}
    assert med["giraffe"] > med["sports ball"] and med["bed"] > med["spoon"], "size sanity check failed"
    v = sorted(med[o] for o in objects)
    t1, t2 = v[len(v) // 3], v[2 * len(v) // 3]
    size = {o: ("small" if med[o] < t1 else "medium" if med[o] < t2 else "large") for o in objects}
    return dict(median_rel_box_area={o: round(med[o], 6) for o in sorted(objects)}, n_boxes={o: len(rel[o]) for o in sorted(objects)},
                cut_t1=t1, cut_t2=t2, type={o: size[o] for o in sorted(objects)},
                excluded_val2017_images=len(pope_coco_ids & set(area)), annotation_sha256=hashlib.sha256(ANN.read_bytes()).hexdigest())


def main():
    rows = load_rows()
    assert len(rows) == 9000, len(rows)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    # images
    img_bytes = {}
    for r in rows:
        b = r["image"]["bytes"]
        prev = img_bytes.setdefault(r["image_source"], b)
        assert prev == b, f"two different images for {r['image_source']}"
    ids = sorted(img_bytes)
    assert len(ids) == 500
    u = dict(zip(ids, np.random.default_rng(SPLIT_STREAM).random(len(ids))))
    sc_md5 = {json.loads(l)["md5"]: json.loads(l)["image_id"] for l in open(SC_IMAGES)}
    from PIL import Image
    import io
    images, overlap = [], []
    for i in ids:
        b = img_bytes[i]
        p = IMG_DIR / f"{i}.jpg"
        if not p.exists():
            p.write_bytes(b)
        assert p.read_bytes() == b
        h = hashlib.md5(b).hexdigest()
        if h in sc_md5:
            overlap.append(dict(image_id=i, sugarcrepe_image=sc_md5[h]))
        with Image.open(io.BytesIO(b)) as im:
            w, hh = im.size
        split = "DEV" if u[i] < 0.4 else ("CAL" if u[i] < 0.7 else "TEST")
        images.append(dict(image_id=i, path=str(p.relative_to(ROOT)), md5=h, width=w, height=hh, u=round(float(u[i]), 6),
                           split=split, pilot_dev=bool(u[i] < 0.3), dev_band=("pilot" if u[i] < 0.3 else "rehearsal" if u[i] < 0.4 else None)))
    by_img = {x["image_id"]: x for x in images}
    # unique (image, object) pairs
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["image_source"], obj_of(r["question"]))].append(r)
    objects = sorted({o for _, o in groups})
    assert len(objects) == 79, len(objects)
    pope_coco_ids = {int(i.split("_")[-1]) for i in ids}
    sizes = size_table(objects, pope_coco_ids)
    questions = []
    for (img, o), rs in sorted(groups.items()):
        answers = {r["answer"].strip().lower() for r in rs}
        assert len(answers) == 1 and answers <= {"yes", "no"}, (img, o, answers)
        good = [r for r in rs if "imange" not in r["question"]]
        canon = (good or rs)[0]["question"]
        meta = by_img[img]
        questions.append(dict(qid=f"pp-{img[-12:]}-{o.replace(' ', '_')}", source="pope", image_id=img, question=canon, object=o,
                              correct=answers.pop(), size_type=sizes["type"][o], settings=sorted({r["category"] for r in rs}),
                              pope_ids=sorted(r["id"] for r in rs), typo_only=not good, split=meta["split"],
                              pilot_dev=meta["pilot_dev"], dev_band=meta["dev_band"]))
    assert len({q["qid"] for q in questions}) == len(questions)
    with open(OUT / "pope_images.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in images)
    with open(OUT / "pope_questions.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in questions)
    cnt = lambda items, key: dict(sorted(collections.Counter(key(x) for x in items).items()))
    meta = dict(rules_version=RULES_VERSION, split_stream=f"default_rng({SPLIT_STREAM})",
                split_rule="DEV u<0.4 (pilot u<0.3 for paths, 0.3<=u<0.4 rehearsal), CAL 0.4<=u<0.7, TEST u>=0.7",
                unit="unique (image, queried object); canonical text = correctly spelled question when available",
                question_regex=QUESTION_RE.pattern, size_rule=__doc__.split("  - Type: ")[1].split("  - Split")[0].strip(),
                size_table=sizes, sugarcrepe_md5_overlap=overlap, n_rows=len(rows), n_images=len(images), n_questions=len(questions),
                n_typo_only=sum(q["typo_only"] for q in questions), yes_share=round(sum(q["correct"] == "yes" for q in questions) / len(questions), 4),
                images_per_split=cnt(images, lambda x: x["split"]), questions_per_split=cnt(questions, lambda x: x["split"]),
                dev_bands=cnt([x for x in images if x["dev_band"]], lambda x: x["dev_band"]),
                questions_per_type=cnt(questions, lambda x: x["size_type"]),
                type_images_per_split={t: cnt({(q["image_id"], q["split"]) for q in questions if q["size_type"] == t}, lambda z: z[1])
                                       for t in ("small", "medium", "large")},
                source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(SRC.glob("test-*.parquet"))})
    (OUT / "pope_meta.json").write_text(json.dumps(meta, indent=1))
    show = {k: v for k, v in meta.items() if k not in ("size_table", "source_sha256", "size_rule")}
    show["cuts"] = (round(sizes["cut_t1"], 5), round(sizes["cut_t2"], 5))
    show["type_counts_objects"] = dict(collections.Counter(sizes["type"].values()))
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    sys.exit(main())
