"""Turn the Hugging Face COCO val2017 mirror into the standard COCO layout.

Writes data/val2017/*.jpg (the SugarCrepe and POPE images) and data/annotations/instances_val2017.json (the boxes that
data_provider/manifest_pope.py uses for its object-size table). Needs pyarrow.

    python data_provider/coco_export.py
"""
import glob
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parent.parent / "data"
IMG = DATA / "val2017"
ANN = DATA / "annotations"


def main():
    IMG.mkdir(exist_ok=True)
    ANN.mkdir(exist_ok=True)

    # images and boxes (the mirror stores boxes as x_min, y_min, x_max, y_max)
    anns, images = [], []
    for f in sorted(glob.glob(str(DATA / "hf" / "val-*.parquet"))):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
            for r in batch.to_pylist():
                iid = r["image_id"]
                path = IMG / f"{iid:012d}.jpg"
                if not path.exists():
                    path.write_bytes(r["image"]["bytes"])
                images.append({"id": iid, "width": r["width"], "height": r["height"]})
                o = r["objects"]
                for bid, cat, (x0, y0, x1, y1), area in zip(o["bbox_id"], o["category"], o["bbox"], o["area"]):
                    anns.append({"id": bid, "image_id": iid, "category_id": cat, "iscrowd": 0,
                                 "bbox": [x0, y0, x1 - x0, y1 - y0], "area": area})
    (ANN / "instances_val2017.json").write_text(json.dumps({"images": images, "annotations": anns}))

    # SugarCrepe images that the mirror lacks (val2017 images without any box): fetch them directly
    need = set()
    for f in glob.glob(str(DATA / "sugarcrepe" / "*.json")):
        for ex in json.loads(Path(f).read_text()).values():
            need.add(ex["filename"])
    missing = [n for n in sorted(need) if not (IMG / n).exists()]

    def fetch(name):
        subprocess.run(["curl", "-sSL", "--retry", "3", "-o", str(IMG / name),
                        f"http://images.cocodataset.org/val2017/{name}"], check=True)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(fetch, missing))
    print(f"{len(images)} mirrored images, {len(anns)} boxes, fetched {len(missing)} extra")


if __name__ == "__main__":
    main()
