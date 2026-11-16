"""DEV paired runs on SugarCrepe (E2): the cheap pass (base view) and the full pass (native AnyRes) on every question of
the plan. The candidate thresholds of the final study are built from these runs (exp/exp_protocol.py).

    python run.py --task_name plan --dataset sugarcrepe                 # freeze the question plan (no model, no answers)
    python run.py --task_name paired --dataset sugarcrepe               # the paired runs (resumes)
    (or bash scripts/sugarcrepe_paired.sh, which builds and extends the plan once and then runs it)

Plan (answer-blind, fixed seeds). pilot_dev images (u < 0.3 in the canonical manifest) are split by image into
internal train / tune / val = 60/20/20 (seed 20261003); 50 images drawn with seed 20261002 for an early engineering
check go to train. tune and val take one question per (image, observable group) (seed 20261004); train fills the rest
of the 1,000-question budget, several questions per image allowed. Run order is a seeded permutation (20261005), so any
prefix is a random answer-blind subset of the plan. extend_plan later appended the rest of pilot_dev (2,322 in all).

Per question: answers, margins and log-probabilities of both paths; stage times with CUDA synchronisation
between stages (base preprocess, base encode, router features, cheap prefill, margin; full preprocess, full
forward, full score); geometry. The run also saves the 1024-d mean base-view tower feature (features/<qid>.npy) for a
router tried during development; the final study does not use it.
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
from root import E2_DIR as E2, MANIFEST_DIR as MANIFESTS, results_dir
from utils.tools import check_run_config, read_jsonl

N_PLAN = 1000
SEEDS = dict(internal_split=20261003, tune_val_pick=20261004, order=20261005, path_order=20261006)
EXTENSION_SEED = 20261007  # order of the questions appended by extend_plan (kept out of SEEDS: run_config is frozen)
GROUPS = ["replace", "add_remove", "reorder", "other"]


SMOKE_SEED = 20261002  # the 50 images of the early engineering check, kept in train
SMOKE_QUOTA = {"replace_obj": 8, "replace_att": 7, "replace_rel": 7, "swap_obj": 7, "swap_att": 7, "add_obj": 7, "add_att": 7}


def _smoke_images():
    """The 50 pilot_dev images of the early engineering check: distinct images drawn in a seeded order, with a quota
    per original SugarCrepe category (answer-blind)."""
    pool = [q for q in read_jsonl(MANIFESTS / "sugarcrepe_questions.jsonl") if q["pilot_dev"] and q["split"] == "DEV"]
    order = np.random.default_rng(SMOKE_SEED).permutation(len(pool))
    left, used = dict(SMOKE_QUOTA), set()
    for i in order:
        q = pool[i]
        if left.get(q["category"], 0) > 0 and q["image_id"] not in used:
            used.add(q["image_id"])
            left[q["category"]] -= 1
    return used


def _pool_and_parts():
    """pilot_dev questions and the internal train / tune / val part of each image (seeded, answer-blind)."""
    qs = read_jsonl(MANIFESTS / "sugarcrepe_questions.jsonl")
    pool = [q for q in qs if q["pilot_dev"] and q["split"] == "DEV"]
    smoke_imgs = _smoke_images()
    imgs = sorted({q["image_id"] for q in pool})
    v = dict(zip(imgs, np.random.default_rng(SEEDS["internal_split"]).random(len(imgs))))
    part = {im: ("train" if (im in smoke_imgs or v[im] < 0.6) else ("tune" if v[im] < 0.8 else "val")) for im in imgs}
    return pool, part


def make_plan(exp_dir):
    pool, part = _pool_and_parts()
    smoke_imgs = _smoke_images()
    imgs = sorted({q["image_id"] for q in pool})
    rng = np.random.default_rng(SEEDS["tune_val_pick"])
    by_img_grp = collections.defaultdict(list)
    for q in pool:
        by_img_grp[(q["image_id"], q["obs_group"])].append(q)
    plan = []
    for (im, g), lst in sorted(by_img_grp.items()):
        if part[im] in ("tune", "val"):
            plan.append(dict(lst[int(rng.integers(len(lst)))], internal=part[im]))
    train_pool = [q for q in pool if part[q["image_id"]] == "train"]
    n_train = max(0, N_PLAN - len(plan))
    pick = rng.permutation(len(train_pool))[:n_train]
    plan += [dict(train_pool[i], internal="train") for i in sorted(pick)]
    order = np.random.default_rng(SEEDS["order"]).permutation(len(plan))
    plan = [dict(plan[i], order=k) for k, i in enumerate(order)]
    exp_dir.mkdir(parents=True, exist_ok=True)
    if (exp_dir / "plan.jsonl").exists():
        raise SystemExit("plan.jsonl exists: the plan is frozen; delete it deliberately to re-plan")
    with open(exp_dir / "plan.jsonl", "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in plan)
    cnt = lambda key: dict(sorted(collections.Counter(key(q) for q in plan).items()))
    meta = dict(seeds=SEEDS, n_plan=len(plan), pilot_dev_images=len(imgs), smoke_images_forced_to_train=len(smoke_imgs),
                images_per_internal=dict(collections.Counter(part.values())), questions_per_internal=cnt(lambda q: q["internal"]),
                internal_by_group=cnt(lambda q: f"{q['internal']}|{q['obs_group']}"), categories=cnt(lambda q: q["category"]))
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


@torch.inference_mode()
def run_paired(exp_dir, limit):
    from exp.sugarcrepe_inputs import prepare
    from models.llava_next import MODEL, QUESTION, REVISION, NextPaths
    from utils.tools import sync
    plan = read_jsonl(exp_dir / "plan.jsonl")
    plan.sort(key=lambda q: q["order"])
    if limit:
        plan = plan[:limit]
    images = {x["image_id"]: x for x in read_jsonl(MANIFESTS / "sugarcrepe_images.jsonl")}
    traces_p = exp_dir / "traces.jsonl"
    done = {json.loads(l)["qid"] for l in open(traces_p)} if traces_p.exists() else set()
    todo = [q for q in plan if q["qid"] not in done]
    print(f"results folder {exp_dir}: plan {len(plan)}, done {len(done)}, to run {len(todo)}", flush=True)
    if not todo:
        summarize(traces_p)
        return
    P = NextPaths("cuda:0")
    cfg = dict(model=MODEL, revision=REVISION, dtype="float16", attn_implementation=P.attn, batch_size=1,
               device=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__,
               python=platform.python_version(), question=QUESTION, a_token_ids=P.a_ids, b_token_ids=P.b_ids,
               seeds=SEEDS, timing="per-stage, CUDA-synchronised between stages; image decode excluded (io_ms)",
               rho="0: the full path recomputes everything", certificate="none (DEV feasibility)")
    check_run_config(exp_dir / "run_config.json", cfg)
    (exp_dir / "features").mkdir(exist_ok=True)
    t = time.perf_counter
    for q in todo[:5]:  # warm-up, not recorded
        it = prepare(P, q, images)
        packed, _ = P.base_encode(P.base_pixel(it["img"]))
        P.score(P.prefill(packed, it["prompt"]).logits[0, -1])
        P.score(P.full_native(it["img"], it["prompt"])[0])
    rng = random.Random(SEEDS["path_order"])
    with open(traces_p, "a") as f:
        for k, q in enumerate(todo):
            it = prepare(P, q, images)
            rec = dict(qid=q["qid"], internal=q["internal"], order=q["order"], image_id=q["image_id"], category=q["category"],
                       obs_group=q["obs_group"], correct=q["correct"], width=images[q["image_id"]]["width"],
                       height=images[q["image_id"]]["height"], aspect=it["aspect"], grid=it["grid"], len_a=it["len_a"],
                       len_b=it["len_b"], io_ms=round(it["io_ms"], 3))
            first = rng.choice(["base", "full"])
            rec["first_path"] = first
            for path in ([first, "full" if first == "base" else "base"]):
                if path == "base":
                    sync(); a = t()
                    pix = P.base_pixel(it["img"]); sync(); b = t()
                    packed, tmean = P.base_encode(pix); sync(); c = t()
                    feat = tmean.cpu().numpy().astype(np.float32); sync(); d = t()
                    logits = P.prefill(packed, it["prompt"]).logits[0, -1]; sync(); e = t()
                    ans, m, la, lb = P.score(logits); sync(); g = t()
                    np.save(exp_dir / "features" / f"{q['qid']}.npy", feat)
                    rec.update(base_answer=ans, base_margin=m, base_logp_a=la, base_logp_b=lb,
                               n_base_tokens=int(packed.shape[0]),
                               ms_S_pre=(b - a) * 1e3, ms_S_enc=(c - b) * 1e3, ms_S_feat=(d - c) * 1e3,
                               ms_P=(e - d) * 1e3, ms_M=(g - e) * 1e3)
                else:
                    sync(); a = t()
                    inputs = P.full_inputs(it["img"], it["prompt"]); sync(); b = t()
                    logits = P.model(**inputs, use_cache=False).logits[0, -1]; sync(); c = t()
                    ans, m, la, lb = P.score(logits); sync(); d = t()
                    rec.update(full_answer=ans, full_margin=m, full_logp_a=la, full_logp_b=lb,
                               n_full_tokens=int((inputs["input_ids"] == P.img_tok).sum()),
                               n_patches=int(inputs["pixel_values"].shape[1]),
                               ms_H_pre=(b - a) * 1e3, ms_H_fwd=(c - b) * 1e3, ms_H_score=(d - c) * 1e3)
            rec["changed"] = rec["base_answer"] != rec["full_answer"]
            for key in list(rec):
                if key.startswith("ms_"):
                    rec[key] = round(rec[key], 3)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(todo)}", flush=True)
    print("done", flush=True)
    summarize(traces_p)


def summarize(traces_path):
    """Print a short summary of everything recorded so far (goes to the run log)."""
    import numpy as np
    rows = read_jsonl(traces_path)
    if not rows:
        return
    full = np.array([r["ms_H_pre"] + r["ms_H_fwd"] + r["ms_H_score"] for r in rows])
    cheap = np.array([r["ms_S_pre"] + r["ms_S_enc"] + r["ms_P"] + r["ms_M"] for r in rows])
    print(f"Summary of {traces_path}: {len(rows)} questions on {len({r['image_id'] for r in rows})} images")
    print(f"  answer changes (cheap vs full): {np.mean([r['changed'] for r in rows]):.1%}")
    print(f"  accuracy: cheap {np.mean([r['base_answer'] == r['correct'] for r in rows]):.1%}, "
          f"full {np.mean([r['full_answer'] == r['correct'] for r in rows]):.1%}")
    print(f"  mean latency (stage sums): full {full.mean():.1f} ms, cheap {cheap.mean():.1f} ms "
          f"({1 - cheap.mean() / full.mean():.1%} saved)")


class ExpPairedSugarCrepe(ExpBasic):
    def __init__(self, args):
        super().__init__(args)
        self.exp_dir = results_dir(args.exp_dir, E2)

    def plan(self):
        make_plan(self.exp_dir)

    def extend(self):
        extend_plan(self.exp_dir)

    def run(self):
        run_paired(self.exp_dir, self.args.limit)
