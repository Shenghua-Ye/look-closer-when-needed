"""Input of one SugarCrepe request: the decoded image, the A/B prompt and a few request features, with the time spent
reading the image (io_ms, excluded from every latency)."""
import time

from PIL import Image
from transformers.image_processing_utils import select_best_resolution

from root import ROOT_DIR as ROOT


def prepare(P, q, images):
    meta = images[q["image_id"]]
    t0 = time.perf_counter()
    img = Image.open(ROOT / meta["path"]).convert("RGB")
    img.load()
    io_ms = (time.perf_counter() - t0) * 1000
    grid = select_best_resolution((meta["height"], meta["width"]), P.model.config.image_grid_pinpoints)
    return dict(q=q, img=img, prompt=P.prompt(q["option_a"], q["option_b"]), obs_group=q["obs_group"],
                len_a=len(q["option_a"].split()), len_b=len(q["option_b"].split()),
                aspect=meta["width"] / meta["height"], grid=[grid[0] / 336, grid[1] / 336], io_ms=io_ms)
