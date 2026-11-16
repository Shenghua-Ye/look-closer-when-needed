"""Export the TextVQA validation parquet (lmms-lab/textvqa) into plain files the model environment can read:
data/textvqa/images/<image_id>.jpg and data/textvqa/questions_val.jsonl. Needs pyarrow (the `cpg` environment).

    python data_provider/textvqa_export.py
"""
import glob
import json

import pyarrow.parquet as pq

from root import DATA_DIR

SRC = DATA_DIR / "hf" / "textvqa" / "data"
OUT = DATA_DIR / "textvqa"


def export():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    for f in sorted(glob.glob(str(SRC / "validation-*.parquet"))):
        for r in pq.read_table(f).to_pylist():
            if r["image_id"] not in seen:
                seen.add(r["image_id"])
                img = r["image"]
                (OUT / "images" / f"{r['image_id']}.jpg").write_bytes(img["bytes"] if isinstance(img, dict) else img)
            rows.append(dict(question_id=r["question_id"], image_id=r["image_id"], question=r["question"],
                             answers=r["answers"], width=r["image_width"], height=r["image_height"],
                             image_classes=r.get("image_classes"), set_name=r.get("set_name")))
    with open(OUT / "questions_val.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print("questions", len(rows), "images", len(seen))


if __name__ == "__main__":
    export()
