"""DEV paired runs on TextVQA (E5): cheap (base view) and full (native AnyRes) short-answer generation on the pilot_dev
questions of the plan. The candidate thresholds of the final study are built from these runs (exp/exp_protocol.py).

    python run.py --task_name plan --dataset textvqa                 # freeze the question plan (answer-blind)
    python run.py --task_name check_decode --dataset textvqa         # decode-loop check on 5 questions
    python run.py --task_name paired --dataset textvqa --limit 100    # first batch
    python run.py --task_name paired --dataset textvqa               # continue (resumes)
    (or bash scripts/textvqa_paired.sh)

Plan: pilot_dev TextVQA images (u < 0.3) split by image into internal train / tune / val = 60/20/20
(seed 20261013); tune and val take one question per (image, question type) (seed 20261014); train fills the
1,000-question budget; run order is a seeded permutation (20261015). extend_plan later appended the rest of pilot_dev
(1,466 in all). The run also saves the mean base-view tower feature (features/<qid>.npy) for a router tried during
development; the final study does not use it.

Cheap path: base view, prefill, first-position margin (top-1 minus top-2 probability), then greedy decoding from the
same KV cache (what a probe does after accepting). Full path: native AnyRes `generate` (the reference).
Greedy, max_new_tokens = 32, fp16, batch 1. Stage times are CUDA-synchronised.
"""
import collections
import json
import platform
import random
import time

import numpy as np
import torch
import transformers

from exp.exp_basic import ExpBasic
from root import E5_DIR as E5, MANIFEST_DIR as MANIFESTS, ROOT_DIR, results_dir
from utils.tools import check_run_config, read_jsonl

ROOT = ROOT_DIR
N_PLAN = 1000
SEEDS = dict(internal_split=20261013, tune_val_pick=20261014, order=20261015, path_order=20261016)
EXTENSION_SEED = 20261017  # order of the questions appended by extend_plan (kept out of SEEDS: run_config is frozen)
TYPES = ["reading", "counting", "existence", "attribute", "relation", "other"]


def _pool_and_parts():
    """pilot_dev questions and the internal train / tune / val part of each image (seeded, answer-blind)."""
    qs = [q for q in read_jsonl(MANIFESTS / "textvqa_questions.jsonl") if q["pilot_dev"] and q["split"] == "DEV"]
    imgs = sorted({q["image_id"] for q in qs})
    v = dict(zip(imgs, np.random.default_rng(SEEDS["internal_split"]).random(len(imgs))))
    part = {im: "train" if v[im] < 0.6 else ("tune" if v[im] < 0.8 else "val") for im in imgs}
    return qs, part


def make_plan(exp_dir):
    qs, part = _pool_and_parts()
    imgs = sorted({q["image_id"] for q in qs})
    rng = np.random.default_rng(SEEDS["tune_val_pick"])
    by = collections.defaultdict(list)
    for q in qs:
        by[(q["image_id"], q["qtype"])].append(q)
    plan = [dict(lst[int(rng.integers(len(lst)))], internal=part[im]) for (im, t), lst in sorted(by.items())
            if part[im] in ("tune", "val")]
    train_pool = [q for q in qs if part[q["image_id"]] == "train"]
    pick = rng.permutation(len(train_pool))[:max(0, N_PLAN - len(plan))]
    plan += [dict(train_pool[i], internal="train") for i in sorted(pick)]
    order = np.random.default_rng(SEEDS["order"]).permutation(len(plan))
    plan = [dict(plan[i], order=k) for k, i in enumerate(order)]
    exp_dir.mkdir(parents=True, exist_ok=True)
    if (exp_dir / "plan.jsonl").exists():
        raise SystemExit("plan.jsonl exists: the plan is frozen")
    with open(exp_dir / "plan.jsonl", "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in plan)
    cnt = lambda key: dict(sorted(collections.Counter(key(q) for q in plan).items()))
    meta = dict(seeds=SEEDS, n_plan=len(plan), pilot_dev_images=len(imgs), images_per_internal=dict(collections.Counter(part.values())),
                questions_per_internal=cnt(lambda q: q["internal"]), internal_by_type=cnt(lambda q: f"{q['internal']}|{q['qtype']}"))
    (exp_dir / "plan_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def extend_plan(exp_dir):
    """Append the rest of pilot_dev to a frozen plan, to cover all of pilot_dev. Existing rows and their order
    stay untouched; new rows get order >= len(plan) from a separate seed. New questions on train images are internal
    'train'; extra questions on tune/val images become 'tune_extra' / 'val_extra', so the original one-question-per-
    (image, group) tune/val sets stay as they were."""
    plan = read_jsonl(exp_dir / "plan.jsonl")
    pool, part = _pool_and_parts()
    bad = [r["qid"] for r in plan if part[r["image_id"]] != r["internal"]]
    if bad:
        raise SystemExit(f"internal split does not reproduce the frozen plan ({len(bad)} rows); not extending")
    have = {r["qid"] for r in plan}
    new = [q for q in pool if q["qid"] not in have]
    if not new:
        print("plan already covers all of pilot_dev"); return
    label = lambda im: part[im] if part[im] == "train" else part[im] + "_extra"
    order = np.random.default_rng(EXTENSION_SEED).permutation(len(new))
    rows = [dict(new[i], internal=label(new[i]["image_id"]), order=len(plan) + k) for k, i in enumerate(order)]
    with open(exp_dir / "plan.jsonl", "a") as f:
        f.writelines(json.dumps(q) + "\n" for q in rows)
    meta = json.loads((exp_dir / "plan_meta.json").read_text())
    meta["extension"] = dict(seed=EXTENSION_SEED, n_added=len(rows), n_plan_total=len(plan) + len(rows),
                             added_by_internal=dict(sorted(collections.Counter(r["internal"] for r in rows).items())))
    (exp_dir / "plan_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["extension"], indent=1))


def load(P, q, images):
    from PIL import Image
    meta = images[q["image_id"]]
    t0 = time.perf_counter()
    img = Image.open(ROOT / meta["path"]).convert("RGB")
    img.load()
    return img, P.prompt_open(q["question"]), (time.perf_counter() - t0) * 1000


@torch.inference_mode()
def check_decode(exp_dir, n=5):
    """The manual decode loop fed with native packed features must reproduce native generate token by token."""
    from models.llava_next import NextPaths
    P = NextPaths("cuda:0")
    plan = sorted(read_jsonl(exp_dir / "plan.jsonl"), key=lambda q: q["order"])[:n]
    images = {x["image_id"]: x for x in read_jsonl(MANIFESTS / "textvqa_images.jsonl")}
    ok = True
    for q in plan:
        img, pr, _ = load(P, q, images)
        ref = P.full_generate(img, pr)
        man = P.greedy_from_embeddings(P.full_packed(img, pr), pr)
        same = ref["tokens"] == man["tokens"]
        ok &= same
        print(q["qid"], repr(ref["text"]), repr(man["text"]), "tokens equal:", same, "margins", round(ref["margin"], 4), round(man["margin"], 4))
    print("CHECK", "PASS" if ok else "FAIL")
    (exp_dir / "decode_check.json").write_text(json.dumps(dict(n=n, all_tokens_equal=ok)))


@torch.inference_mode()
def run_paired(exp_dir, limit):
    from models.llava_next import MAX_NEW, MODEL, OPEN_SUFFIX, REVISION, NextPaths
    from utils.vqa_eval import VERSION, accuracy, normalize
    plan = sorted(read_jsonl(exp_dir / "plan.jsonl"), key=lambda q: q["order"])
    if limit:
        plan = plan[:limit]
    images = {x["image_id"]: x for x in read_jsonl(MANIFESTS / "textvqa_images.jsonl")}
    tp = exp_dir / "traces.jsonl"
    done = {json.loads(l)["qid"] for l in open(tp)} if tp.exists() else set()
    todo = [q for q in plan if q["qid"] not in done]
    print(f"results folder {exp_dir}: plan {len(plan)}, done {len(done)}, to run {len(todo)}", flush=True)
    if not todo:
        summarize(tp)
        return
    P = NextPaths("cuda:0")
    check_run_config(exp_dir / "run_config.json", dict(
        model=MODEL, revision=REVISION, dtype="float16", attn_implementation=P.attn, batch_size=1, greedy=True,
        max_new_tokens=MAX_NEW, prompt_suffix=OPEN_SUFFIX, normalizer=VERSION, device=torch.cuda.get_device_name(0),
        torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(), seeds=SEEDS,
        margin="top-1 minus top-2 softmax probability at the first generation position",
        timing="per stage, CUDA-synchronised; image decode excluded (io_ms)", rho="0"))
    (exp_dir / "features").mkdir(exist_ok=True)
    t = time.perf_counter
    sync = torch.cuda.synchronize
    for q in todo[:3]:  # warm-up
        img, pr, _ = load(P, q, images)
        packed, _ = P.base_encode(P.base_pixel(img))
        P.greedy_from_embeddings(packed, pr)
        P.full_generate(img, pr)
    rng = random.Random(SEEDS["path_order"])
    with open(tp, "a") as f:
        for k, q in enumerate(todo):
            img, pr, io = load(P, q, images)
            meta = images[q["image_id"]]
            rec = dict(qid=q["qid"], internal=q["internal"], order=q["order"], image_id=q["image_id"], qtype=q["qtype"],
                       question=q["question"], width=meta["width"], height=meta["height"], io_ms=round(io, 3))
            first = rng.choice(["base", "full"])
            rec["first_path"] = first
            for path in [first, "full" if first == "base" else "base"]:
                if path == "base":
                    marks = {}
                    sync(); a = t()
                    pix = P.base_pixel(img); sync(); b = t()
                    packed, tmean = P.base_encode(pix); sync(); c = t()
                    feat = tmean.cpu().numpy().astype(np.float32); sync(); d = t()
                    def clock(name):
                        sync(); marks[name] = t()
                    res = P.greedy_from_embeddings(packed, pr, clock=clock)
                    np.save(exp_dir / "features" / f"{q['qid']}.npy", feat)
                    rec.update(base_text=res["text"], base_norm=normalize(res["text"]), base_margin=res["margin"],
                               base_first_token=res["tokens"][0], base_n_new=res["n_new"], base_truncated=res["truncated"],
                               base_acc=accuracy(res["text"], q["answers"]), n_base_tokens=int(packed.shape[0]),
                               ms_S_pre=(b - a) * 1e3, ms_S_enc=(c - b) * 1e3, ms_S_feat=(d - c) * 1e3,
                               ms_P=(marks["prefill"] - d) * 1e3, ms_D=(marks["decode"] - marks["prefill"]) * 1e3)
                else:
                    sync(); a = t()
                    res = P.full_generate(img, pr); sync(); b = t()
                    rec.update(full_text=res["text"], full_norm=normalize(res["text"]), full_margin=res["margin"],
                               full_first_token=res["tokens"][0], full_n_new=res["n_new"], full_truncated=res["truncated"],
                               full_acc=accuracy(res["text"], q["answers"]), n_full_tokens=res["n_visual_tokens"],
                               ms_H=(b - a) * 1e3)
            rec["changed"] = rec["base_norm"] != rec["full_norm"]
            for key in list(rec):
                if key.startswith("ms_"):
                    rec[key] = round(rec[key], 3)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (k + 1) % 25 == 0:
                print(f"{k + 1}/{len(todo)}", flush=True)
    print("done", flush=True)
    summarize(tp)


def summarize(traces_path):
    """Print a short summary of everything recorded so far (goes to the run log)."""
    rows = read_jsonl(traces_path)
    if not rows:
        return
    full = np.array([r["ms_H"] for r in rows])
    cheap = np.array([r["ms_S_pre"] + r["ms_S_enc"] + r["ms_P"] + r["ms_D"] for r in rows])
    print(f"Summary of {traces_path}: {len(rows)} questions on {len({r['image_id'] for r in rows})} images")
    print(f"  answer changes (cheap vs full, VQA-normalised): {np.mean([r['changed'] for r in rows]):.1%}")
    print(f"  VQA accuracy: cheap {np.mean([r['base_acc'] for r in rows]):.1%}, full {np.mean([r['full_acc'] for r in rows]):.1%}")
    print(f"  mean latency (stage sums): full {full.mean():.1f} ms, cheap {cheap.mean():.1f} ms "
          f"({1 - cheap.mean() / full.mean():.1%} saved)")


class ExpPairedTextVQA(ExpBasic):
    def __init__(self, args):
        super().__init__(args)
        self.exp_dir = results_dir(args.exp_dir, E5)

    def plan(self):
        make_plan(self.exp_dir)

    def extend(self):
        extend_plan(self.exp_dir)

    def check(self):
        check_decode(self.exp_dir)

    def run(self):
        run_paired(self.exp_dir, self.args.limit)
