"""POPE online latency on round-0 TEST: always-full, the certified restart policy and the certified resume policy, timed
per request in one session, as exp_f3_timing does for the other two datasets. The summary is exp_f3_timing's.

    python -m exp.exp_pope_timing --dry_run      # CPU: request list, thresholds, replay prediction
    python -m exp.exp_pope_timing                # GPU (one; the others idle): the timed run (resumes)
    python -m exp.exp_pope_timing --summarize    # CPU

Requests: every round-0 TEST question in a seeded order, written before timing. Thresholds: round 0, main
configuration, alpha = 5%, per-type LTT from the POPE certificates (restart arm: restart certificate; resume arm: resume
certificate). Arms (R = 3 repeats, arms in a random order per round; per-request median):
  full     native full pass (processor + forward + forced choice)
  restart  cheap path; accept if m > lambda_g, else the native full pass from scratch
  resume   the cheap path keeping its KV cache; accept as above; else the continuation of exp_pope_paired
Timing boundaries, warm-up and GPU-exclusivity checks as exp_f3_timing (exp/timing_tools.py).
Outputs: results/P3_pope_timing_test/.
"""
import argparse
import json
import random
import statistics
import time
from datetime import datetime

import numpy as np

from exp import pope_common as C
from exp.exp_f3_timing import ARMS, summarize
from exp.timing_tools import exclusivity, keep_kv, own_gpu, warmup_set
from root import RESULTS_DIR, ROOT_DIR
from utils.tools import check_run_config

OUT = RESULTS_DIR / "P3_pope_timing_test"
MAIN = "signal-first_margin_loss-strict"
SEED = 20261107
N_BASE = 576


def thresholds():
    cp = C.P2_DIR / "certificates_pope.json"
    r0 = next(x for x in json.loads(cp.read_text())["rounds"] if x["round"] == 0)
    pick = lambda fb: next(c for c in r0["runs"] if c["config"] == MAIN and c["rule"] == "group_ltt" and c["fallback"] == fb
                           and abs(c["alpha"] - C.ALPHA_MAIN) < 1e-12)["thresholds"]
    return dict(restart=pick("restart"), resume=pick("resume"))


def build():
    test = set(json.loads((C.LOCK_DIR / "splits_pope.json").read_text())["rounds"][0]["TEST"])
    lam = thresholds()
    rows = sorted((r for r in C.read_jsonl(C.P1_DIR / "traces.jsonl") if r["image_id"] in test), key=lambda r: r["qid"])
    order = np.random.default_rng([SEED, C.CODE]).permutation(len(rows))
    reqs = []
    for k, i in enumerate(order):
        r = rows[i]
        S0, Cc, H = r["ms_S_pre"] + r["ms_S_enc"], r["ms_P"] + r["ms_M"], r["ms_H_pre"] + r["ms_H_fwd"] + r["ms_H_score"]
        Hr = r["ms_R_pre"] + r["ms_R_vis"] + r["ms_R_cont"] + r["ms_R_dec"]
        q = dict(k=k, qid=r["qid"], image_id=r["image_id"], group=r["group"], trace_margin=r["base_margin"], ms_replay_full=H)
        for arm, fb in (("restart", H), ("resume", Hr)):
            t = lam[arm][r["group"]]
            q[f"threshold_{arm}"] = t
            if t is None:
                q[f"replay_{arm}"], q[f"ms_replay_{arm}"] = "safe_full", H
            elif r["base_margin"] > t:
                q[f"replay_{arm}"], q[f"ms_replay_{arm}"] = "accept", S0 + Cc
            else:
                q[f"replay_{arm}"], q[f"ms_replay_{arm}"] = "fallback", S0 + Cc + fb
        reqs.append(q)
    return dict(dataset="pope", thresholds=lam, requests=reqs)


def make_arms(P, images):
    from PIL import Image
    from exp.exp_pope_paired import yes_no_scorer
    from exp.exp_reuse import Reuse
    R = Reuse(P)
    score, _ = yes_no_scorer(P)

    def load(q):
        img = Image.open(ROOT_DIR / images[q["image_id"]]["path"]).convert("RGB")
        img.load()
        return img, P.prompt_open(q["question"])

    def full(img, pr, lam=None):
        return dict(answer=score(P.full_native(img, pr)[0])[0])

    def restart(img, pr, lam):
        if lam is None:
            return dict(full(img, pr), action="safe_full")
        packed, _ = P.base_encode(P.base_pixel(img))
        ans, m, _, _ = score(P.prefill(packed, pr).logits[0, -1])
        if m > lam:
            return dict(answer=ans, action="accept", margin=m)
        return dict(full(img, pr), action="fallback", margin=m)

    def resume(img, pr, lam):
        if lam is None:
            return dict(full(img, pr), action="safe_full")
        packed, _ = P.base_encode(P.base_pixel(img))
        with keep_kv(P) as kv:
            out = P.prefill(packed, pr, use_cache=True)
        ans, m, _, _ = score(out.logits[0, -1])
        del out
        if m > lam:
            return dict(answer=ans, action="accept", margin=m)
        pre, post = P._split_ids(pr)
        packed_f, _ = R.crop_features(img, pr, packed)
        o2 = R.continue_prefill(kv.cache, len(pre) + N_BASE, packed_f, post)
        return dict(answer=score(o2.logits[0, -1])[0], action="fallback", margin=m)
    return load, dict(full=full, restart=restart, resume=resume)


def execute(ctx, args, start):
    import torch
    from exp.exp_f1_paired import load_model
    reqs = ctx["requests"]
    OUT.mkdir(parents=True, exist_ok=True)
    rq = OUT / "requests.json"
    frozen = json.loads(json.dumps(ctx))
    if rq.exists() and json.loads(rq.read_text()) != frozen:
        raise SystemExit(f"{rq} holds a different request list")
    if not rq.exists():
        rq.write_text(json.dumps(frozen))
    tp, cpath = OUT / "traces.jsonl", OUT / "gpu_checks.jsonl"
    done = {json.loads(l)["qid"] for l in open(tp)} if tp.exists() else set()
    todo = [q for q in reqs if q["qid"] not in done]
    print(f"{OUT}: {len(reqs)} requests, done {len(done)}, to run {len(todo)}", flush=True)
    if not todo:
        return
    P = load_model()
    sync, reset = torch.cuda.synchronize, torch.cuda.reset_peak_memory_stats
    device = torch.cuda.get_device_name(0)
    check_run_config(OUT / "run_config.json", dict(dataset="pope", device=device, repeats=args.repeats, order_seed=SEED,
                                                   thresholds=ctx["thresholds"]))
    qmeta = {q["qid"]: q for q in C.questions()}
    images = C.images()
    load, arms = make_arms(P, images)
    with open(cpath, "a") as f:
        f.write(json.dumps(dict(start, kind="start", n_done=len(done), forced=args.force, device=device,
                                weights_mib=round(torch.cuda.memory_allocated() / 2 ** 20, 1))) + "\n")
    t = time.perf_counter
    with torch.inference_mode():
        for q in warmup_set(todo, images, P.model.config.image_grid_pinpoints, args.warmup):
            img, pr = load(qmeta[q["qid"]])
            arms["full"](img, pr)
            arms["restart"](img, pr, float("inf"))
            arms["resume"](img, pr, float("inf"))
        sync()
        with open(tp, "a") as f:
            for k, q in enumerate(todo):
                img, pr = load(qmeta[q["qid"]])
                rng = random.Random(f"{SEED}:{q['qid']}")
                runs = []
                for rnd in range(args.repeats):
                    order = list(ARMS)
                    rng.shuffle(order)
                    for arm in order:
                        lam = None if arm == "full" else q[f"threshold_{arm}"]
                        reset()
                        sync(); a = t()
                        res = arms[arm](img, pr, lam)
                        sync(); b = t()
                        res.update(round=rnd, arm=arm, ms=round((b - a) * 1e3, 3), peak_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1))
                        runs.append(res)
                rec = dict(q, runs=runs)
                for arm in ARMS:
                    rr = [x for x in runs if x["arm"] == arm]
                    rec[f"ms_{arm}"] = float(statistics.median(x["ms"] for x in rr))
                    rec[f"peak_mib_{arm}"] = max(x["peak_mib"] for x in rr)
                    rec[f"answer_{arm}"] = rr[0]["answer"]
                    rec[f"stable_{arm}"] = len({x["answer"] for x in rr}) == 1
                    if arm != "full":
                        rec[f"action_{arm}"] = rr[0]["action"]
                        rec[f"decisions_ok_{arm}"] = all(x["action"] == q[f"replay_{arm}"] for x in rr)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if (k + 1) % args.check_every == 0 or k + 1 == len(todo):
                    with open(cpath, "a") as g:
                        g.write(json.dumps(dict(exclusivity(start["own"], during_run=True), own=start["own"],
                                                kind="periodic" if k + 1 < len(todo) else "end", n_done=len(done) + k + 1)) + "\n")
                    print(f"[{datetime.now():%H:%M:%S}] {k + 1}/{len(todo)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--check_every", type=int, default=100)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        summarize(OUT)
        return
    ctx = build()
    rp = {k: np.array([q[f"ms_replay_{k}"] for q in ctx["requests"]]) for k in ARMS}
    print(f"pope: {len(ctx['requests'])} requests; thresholds {ctx['thresholds']}; replay saving restart "
          f"{1 - rp['restart'].sum() / rp['full'].sum():.1%}, resume {1 - rp['resume'].sum() / rp['full'].sum():.1%}; "
          f"timed GPU estimate {a.repeats * sum(v.sum() for v in rp.values()) / 3.6e6:.2f} h", flush=True)
    if a.dry_run:
        return
    own = own_gpu()
    start = dict(exclusivity(own), own=own)
    if not start["ok"]:
        msg = f"GPUs not exclusive: {start.get('other_procs')}, busy {start.get('busy_gpus')}"
        if not a.force:
            raise SystemExit(msg)
        print("WARNING (--force): " + msg, flush=True)
    execute(ctx, a, start)
    summarize(OUT)


if __name__ == "__main__":
    main()
