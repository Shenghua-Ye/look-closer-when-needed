"""F3: online latency on round-0 TEST: always-full, the certified restart policy and the certified resume policy, timed
per request in one session.

    python -m exp.exp_f3_timing --dataset sugarcrepe --dry_run      # CPU: request list, thresholds, replay prediction
    python -m exp.exp_f3_timing --dataset sugarcrepe                 # GPU (one, others idle): the timed run (resumes)
    python -m exp.exp_f3_timing --dataset sugarcrepe --summarize     # CPU: summary.md / summary.json
    (or bash scripts/f3_timing.sh)

Requests: every round-0 TEST question (SugarCrepe 2,268, TextVQA 1,589), in a seeded order; the list (qid, group,
thresholds) is written before timing and never changes (a rerun only adds missing requests).
Thresholds: round 0, main configuration, main alpha, per-group LTT from results/F2_certify_evaluate/certificates.json:
the restart arm uses the restart certificate, the resume arm the resume certificate.
Arms (R = 3 repeats each, the three arms in a random order per round, seeded per request; per-request median):
  full     native full pass (processor + forward + score / generate)
  restart  cheap path; accept if m > lambda_g, else the native full pass from scratch; no certificate -> full
  resume   the same cheap path keeping its KV cache (timing_tools.keep_kv); accept as above; else the resume
           continuation of exp_f1_paired.resume_fallback (the code F1 measured and the certificate covers)
Timing boundaries, warm-up and GPU-exclusivity checks from exp/timing_tools.py (start: decoded image + prompt string in memory,
synchronise; stop: answer on the host, synchronise). Peak memory per call. The replay prediction of every request comes
from F1's stage times (cost model of exp_frontier), so online and replay can be compared.
Summary: mean / p50 / p95 per arm, saving vs always-full with a paired 95% image-cluster bootstrap (2,000 resamples,
seed 20261107), per-group savings, fallback extra time, decision agreement with the replay, answer agreement.
Outputs: results/F3_timing_test/<dataset>/.
"""
import argparse
import json
import random
import statistics
import time
from datetime import datetime

import numpy as np

from exp.exp_f1_paired import load_model, resume_fallback
from exp.timing_tools import _stats, exclusivity, keep_kv, own_gpu, warmup_set
from root import MANIFEST_DIR, RESULTS_DIR
from utils.tools import check_run_config, read_jsonl

OUT_ROOT = RESULTS_DIR / "F3_timing_test"
F1 = RESULTS_DIR / "F1_paired_caltest"
F2 = RESULTS_DIR / "F2_certify_evaluate"
LOCK_DIR = RESULTS_DIR / "F0_protocol_lock"
MAIN = {"sugarcrepe": ("signal-first_margin_loss-strict", 0.05, "base_margin"),
        "textvqa": ("signal-seq_min_margin_loss-strict", 0.10, "seq_min_margin")}
ARMS = ("full", "restart", "resume")
SEED, BOOT, BOOT_SEED = 20261107, 2000, 20261107


def thresholds(dataset):
    certs = json.loads((F2 / "certificates.json").read_text())
    cfg, alpha, _ = MAIN[dataset]
    r0 = next(x for x in certs["datasets"][dataset] if x["round"] == 0)
    pick = lambda fb: next(c for c in r0["runs"] if c["config"] == cfg and c["rule"] == "group_ltt" and c["fallback"] == fb
                           and abs(c["alpha"] - alpha) < 1e-12)["thresholds"]
    return dict(restart=pick("restart"), resume=pick("resume"))


def build(dataset):
    """Request list with thresholds and the replay prediction (F1 stage times) of every request."""
    splits = json.loads((LOCK_DIR / f"splits_{dataset}.json").read_text())
    test = set(splits["rounds"][0]["TEST"])
    lam = thresholds(dataset)
    rows = [r for r in read_jsonl(F1 / dataset / "traces.jsonl") if r["image_id"] in test]
    rows.sort(key=lambda r: r["qid"])
    order = np.random.default_rng([SEED, 1 if dataset == "sugarcrepe" else 2]).permutation(len(rows))
    mkey = MAIN[dataset][2]
    reqs = []
    for k, i in enumerate(order):
        r = rows[i]
        if dataset == "sugarcrepe":
            S0, C, PM, H = r["ms_S_pre"] + r["ms_S_enc"], r["ms_P"] + r["ms_M"], r["ms_P"] + r["ms_M"], r["ms_H_pre"] + r["ms_H_fwd"] + r["ms_H_score"]
        else:
            S0, C, PM, H = r["ms_S_pre"] + r["ms_S_enc"], r["ms_P"] + r["ms_D"], r["ms_P"] + r["ms_D"], r["ms_H"]
        Hr = r["ms_R_pre"] + r["ms_R_vis"] + r["ms_R_cont"] + r["ms_R_dec"]
        q = dict(k=k, qid=r["qid"], image_id=r["image_id"], group=r["group"], trace_margin=r[mkey], ms_replay_full=H)
        for arm, fb in (("restart", H), ("resume", Hr)):
            t = lam[arm][r["group"]]
            q[f"threshold_{arm}"] = t
            if t is None:
                q[f"replay_{arm}"], q[f"ms_replay_{arm}"] = "safe_full", H
            elif r[mkey] > t:
                q[f"replay_{arm}"], q[f"ms_replay_{arm}"] = "accept", S0 + C
            else:
                q[f"replay_{arm}"], q[f"ms_replay_{arm}"] = "fallback", S0 + PM + fb
        reqs.append(q)
    return dict(dataset=dataset, thresholds=lam, requests=reqs)


def make_arms(P, dataset, images):
    from exp.exp_reuse import Reuse
    R = Reuse(P)
    if dataset == "sugarcrepe":
        from exp.sugarcrepe_inputs import prepare

        def load(q):
            it = prepare(P, q, images)
            return it["img"], it["prompt"]

        def full(img, pr, lam=None):
            return dict(answer=P.score(P.full_native(img, pr)[0])[0])

        def restart(img, pr, lam):
            if lam is None:
                return dict(full(img, pr), action="safe_full")
            packed, _ = P.base_encode(P.base_pixel(img))
            ans, m, _, _ = P.score(P.prefill(packed, pr).logits[0, -1])
            if m > lam:
                return dict(answer=ans, action="accept", margin=m)
            return dict(full(img, pr), action="fallback", margin=m)

        def resume(img, pr, lam):
            if lam is None:
                return dict(full(img, pr), action="safe_full")
            packed, _ = P.base_encode(P.base_pixel(img))
            with keep_kv(P) as kv:
                out = P.prefill(packed, pr, use_cache=True)
            ans, m, _, _ = P.score(out.logits[0, -1])
            del out
            if m > lam:
                return dict(answer=ans, action="accept", margin=m)
            ans_r, info = resume_fallback(P, R, dataset, img, pr, kv.cache, packed, None)
            return dict(answer=ans_r, action="fallback", margin=m)
    else:
        from exp.exp_paired_textvqa import load as load_question
        from exp.exp_signals import decode_with_signals

        def load(q):
            img, pr, _ = load_question(P, q, images)
            return img, pr

        def full(img, pr, lam=None):
            return dict(answer=P.full_generate(img, pr)["text"])

        def restart(img, pr, lam):
            if lam is None:
                return dict(full(img, pr), action="safe_full")
            packed, _ = P.base_encode(P.base_pixel(img))
            text, toks, sig = decode_with_signals(P, packed, pr)
            if sig["seq_min_margin"] > lam:
                return dict(answer=text, action="accept", margin=sig["seq_min_margin"])
            return dict(full(img, pr), action="fallback", margin=sig["seq_min_margin"])

        def resume(img, pr, lam):
            if lam is None:
                return dict(full(img, pr), action="safe_full")
            packed, _ = P.base_encode(P.base_pixel(img))
            with keep_kv(P) as kv:
                text, toks, sig = decode_with_signals(P, packed, pr)
            if sig["seq_min_margin"] > lam:
                return dict(answer=text, action="accept", margin=sig["seq_min_margin"])
            ans_r, info = resume_fallback(P, R, dataset, img, pr, kv.cache, packed, toks)
            return dict(answer=ans_r, action="fallback", margin=sig["seq_min_margin"])
    return load, dict(full=full, restart=restart, resume=resume)


def execute(ctx, args, out_dir, start):
    import torch
    from utils.vqa_eval import normalize
    ds, reqs = ctx["dataset"], ctx["requests"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rq = out_dir / "requests.json"
    frozen = json.loads(json.dumps(ctx))
    if rq.exists() and json.loads(rq.read_text()) != frozen:
        raise SystemExit(f"{rq} holds a different request list")
    if not rq.exists():
        rq.write_text(json.dumps(frozen))
    tp, cp = out_dir / "traces.jsonl", out_dir / "gpu_checks.jsonl"
    done = {json.loads(l)["qid"] for l in open(tp)} if tp.exists() else set()
    todo = [q for q in reqs if q["qid"] not in done]
    print(f"{out_dir}: {len(reqs)} requests, done {len(done)}, to run {len(todo)}", flush=True)
    if not todo:
        return
    P = load_model()
    sync, reset = torch.cuda.synchronize, torch.cuda.reset_peak_memory_stats
    device = torch.cuda.get_device_name(0)
    cfg = dict(dataset=ds, device=device, repeats=args.repeats, order_seed=SEED, thresholds=ctx["thresholds"])
    check_run_config(out_dir / "run_config.json", cfg)
    qmeta = {q["qid"]: q for q in read_jsonl(MANIFEST_DIR / f"{ds}_questions.jsonl")}
    images = {x["image_id"]: x for x in read_jsonl(MANIFEST_DIR / f"{ds}_images.jsonl")}
    load, arms = make_arms(P, ds, images)
    norm = normalize if ds == "textvqa" else (lambda a: a)
    with open(cp, "a") as f:
        f.write(json.dumps(dict(start, kind="start", n_done=len(done), forced=args.force, device=device,
                                weights_mib=round(torch.cuda.memory_allocated() / 2 ** 20, 1))) + "\n")
    t = time.perf_counter
    with torch.inference_mode():
        warm = warmup_set(todo, images, P.model.config.image_grid_pinpoints, args.warmup)
        for q in warm:
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
                        res.update(round=rnd, arm=arm, ms=round((b - a) * 1e3, 3),
                                   peak_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1))
                        res["answer"] = norm(res["answer"])
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
                    with open(cp, "a") as g:
                        g.write(json.dumps(dict(exclusivity(start["own"], during_run=True), own=start["own"],
                                                kind="periodic" if k + 1 < len(todo) else "end", n_done=len(done) + k + 1)) + "\n")
                    print(f"[{datetime.now():%H:%M:%S}] {k + 1}/{len(todo)}", flush=True)


def boot(rows, seed):
    """Paired image-cluster bootstrap of the saving of each policy arm, and resume minus restart."""
    imgs, inv = np.unique([r["image_id"] for r in rows], return_inverse=True)
    s = {a: np.bincount(inv, [r[f"ms_{a}"] for r in rows]) for a in ARMS}
    rng = np.random.default_rng(seed)
    out = {"restart": [], "resume": [], "resume_minus_restart": []}
    for _ in range(BOOT):
        w = np.bincount(rng.integers(len(imgs), size=len(imgs)), minlength=len(imgs))
        f = w @ s["full"]
        a, b = 1 - (w @ s["restart"]) / f, 1 - (w @ s["resume"]) / f
        out["restart"].append(a); out["resume"].append(b); out["resume_minus_restart"].append(b - a)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def summarize(out_dir):
    meta = json.loads((out_dir / "requests.json").read_text())
    rows = read_jsonl(out_dir / "traces.jsonl")
    checks = read_jsonl(out_dir / "gpu_checks.jsonl")
    ds, n, N = meta["dataset"], len(rows), len(meta["requests"])
    on = {a: np.array([r[f"ms_{a}"] for r in rows]) for a in ARMS}
    rp = {a: np.array([r[f"ms_replay_{a}"] for r in rows]) for a in ARMS}
    sav = {a: float(1 - on[a].sum() / on["full"].sum()) for a in ("restart", "resume")}
    sav_rp = {a: float(1 - rp[a].sum() / rp["full"].sum()) for a in ("restart", "resume")}
    ci = boot(rows, BOOT_SEED)
    groups = list(meta["thresholds"]["restart"])
    by_g = {g: {a: float(1 - sum(r[f"ms_{a}"] for r in rows if r["group"] == g) / max(1e-9, sum(r["ms_full"] for r in rows if r["group"] == g)))
                for a in ("restart", "resume")} for g in groups if any(r["group"] == g for r in rows)}
    over = {}
    for a in ("restart", "resume"):
        idx = [i for i, r in enumerate(rows) if r[f"action_{a}"] == "fallback"]
        over[a] = _stats(on[a][idx] - on["full"][idx])
    dec = {a: sum(r[f"decisions_ok_{a}"] for r in rows) for a in ("restart", "resume")}
    loss = {a: sum(r[f"answer_{a}"] != r["answer_full"] for r in rows) for a in ("restart", "resume")}
    mem = {a: max(r[f"peak_mib_{a}"] for r in rows) for a in ARMS}
    gchecks = [c for c in checks if c["kind"] in ("start", "periodic", "end")]
    excl = all(c["ok"] for c in gchecks) and not any(c.get("forced") for c in checks if c["kind"] == "start")
    acts = {a: {x: sum(r[f"action_{a}"] == x for r in rows) for x in ("accept", "fallback", "safe_full")} for a in ("restart", "resume")}
    f = lambda st: "n/a" if st is None else f"{st['mean']:.1f} / {st['p50']:.1f} / {st['p95']:.1f}"
    L = [f"# F3 online latency, round-0 TEST ({ds})", "",
         f"{n} of {N} requests; R = {rows[0]['runs'][-1]['round'] + 1} timed repeats per arm, random arm order per round; per-request "
         f"median. Device {checks[0].get('device')}. GPU exclusivity: {'held at every check' if excl else 'NOT held at every check'} "
         f"({len(gchecks)} checks).", "",
         "| arm | mean / p50 / p95 ms (online) | replay (F1 stage times) | saving online [95% image bootstrap] | saving replay | peak MiB |",
         "|---|---|---|---|---|---|",
         f"| always full | {f(_stats(on['full']))} | {f(_stats(rp['full']))} | | | {mem['full']:.0f} |"]
    for a in ("restart", "resume"):
        L.append(f"| {a} | {f(_stats(on[a]))} | {f(_stats(rp[a]))} | {sav[a]:.1%} [{ci[a][0]:.1%}, {ci[a][1]:.1%}] | {sav_rp[a]:.1%} | {mem[a]:.0f} |")
    L += ["", f"- Resume minus restart saving: {sav['resume'] - sav['restart']:+.1%} [{ci['resume_minus_restart'][0]:+.1%}, {ci['resume_minus_restart'][1]:+.1%}] (paired).",
          "- Actions: " + "; ".join(f"{a}: " + ", ".join(f"{k} {v}" for k, v in acts[a].items()) for a in acts) + ".",
          "- Decisions equal to the F1 replay (every repeat): " + ", ".join(f"{a} {dec[a]}/{n}" for a in dec) + ".",
          "- Fallback extra ms over always-full on the same request (mean / p50 / p95): " + "; ".join(f"{a} {f(over[a])} (max {over[a]['max']:.1f})" for a in over if over[a]) + ".",
          "- Returned answer != this session's native answer: " + ", ".join(f"{a} {loss[a]}/{n}" for a in loss) + ".",
          "- Same answer in every repeat: " + ", ".join(f"{a} {np.mean([r[f'stable_{a}'] for r in rows]):.1%}" for a in ARMS) + ".",
          "- Saving per group (online): " + "; ".join(f"{g}: restart {v['restart']:.1%}, resume {v['resume']:.1%}" for g, v in by_g.items()) + "."]
    (out_dir / "summary.md").write_text("\n".join(L) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(dict(n=n, saving=sav, saving_replay=sav_rp, ci=ci, by_group=by_g,
                                                         latency={a: _stats(on[a]) for a in ARMS}, latency_replay={a: _stats(rp[a]) for a in ARMS},
                                                         fallback_extra=over, decisions_ok=dec, losses=loss, peak_mib=mem,
                                                         exclusive=excl, actions=acts), indent=1))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(MAIN))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--check_every", type=int, default=100)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out_suffix", default="", help="e.g. _shared: a non-exclusive run kept apart from the exclusive one")
    a = ap.parse_args()
    out_dir = OUT_ROOT / (a.dataset + a.out_suffix)
    if a.summarize:
        summarize(out_dir)
        return
    ctx = build(a.dataset)
    rp = {k: np.array([q[f"ms_replay_{k}"] for q in ctx["requests"]]) for k in ARMS}
    print(f"{a.dataset}: {len(ctx['requests'])} requests; thresholds {ctx['thresholds']}; replay saving restart "
          f"{1 - rp['restart'].sum() / rp['full'].sum():.1%}, resume {1 - rp['resume'].sum() / rp['full'].sum():.1%}; "
          f"timed GPU estimate {a.repeats * sum(v.sum() for v in rp.values()) / 3.6e6:.2f} h")
    if a.dry_run:
        return
    own = own_gpu()
    start = dict(exclusivity(own), own=own)
    if not start["ok"]:
        msg = f"GPUs not exclusive: {start.get('other_procs')}, busy {start.get('busy_gpus')}"
        if not a.force:
            raise SystemExit(msg)
        print("WARNING (--force): " + msg, flush=True)
    execute(ctx, a, out_dir, start)
    summarize(out_dir)


if __name__ == "__main__":
    main()
