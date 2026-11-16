"""POPE paired runs: native full, cheap (base view) and forced resume on every question of one stage. One script for DEV
and for CAL/TEST, so the DEV paths and the final runs share their code.

    python -m exp.exp_pope_paired --regress_sc 30                 # GPU: this runner with the A/B scorer must reproduce
                                                                  #      exp_f1_paired's SugarCrepe passes (pilot_dev)
    python -m exp.exp_pope_paired --stage dev [--shard k --n_shards n]      # GPU: all DEV pairs (pilot + rehearsal band)
    python -m exp.exp_pope_paired --stage dev --merge                       # CPU
    python -m exp.exp_pope_paired --stage caltest [--shard k --n_shards n]  # GPU: refuses without the POPE lock
    python -m exp.exp_pope_paired --stage caltest --merge

Per question (fields named as SugarCrepe's F1 rows, so the main study's F2 functions apply unchanged):
  full    processor + model forward on the native AnyRes input; forced choice at the answer position (ms_H_pre, ms_H_fwd,
          ms_H_score)
  cheap   base_pixel, base_encode, prefill with the KV cache kept (exp_f1_paired.TimedKeepKV), forced choice (ms_S_pre,
          ms_S_enc, ms_P, ms_M)
  resume  forced fallback from that cache: exp_reuse.Reuse.crop_features + continue_prefill, forced choice (ms_R_pre,
          ms_R_vis, ms_R_cont, ms_R_dec)
Forced choice: answer = "yes" if logsumexp log p(YES variants) > logsumexp log p(NO variants) else "no"; margin =
|difference| (exp/pope_common.py YES / NO). Prompt: the question + LLaVA's short-answer suffix (NextPaths.prompt_open).
The full and the cheap + resume passes run in a seeded random order per question; image decoding is excluded (io_ms).
DEV only, outside the clock: the full path's free greedy answer (4 tokens) to measure agreement with the forced choice.
Sealing: the caltest stage writes rows only; nothing computed from CAL/TEST answers is printed.
"""
import argparse
import json
import platform
import random
import time
from datetime import datetime

import numpy as np

from exp import pope_common as C
from root import E2_DIR, MANIFEST_DIR, ROOT_DIR
from utils.tools import check_run_config

N_BASE = 576


def make_plan(stage):
    qs = [q for q in C.questions() if (q["split"] == "DEV") == (stage == "dev")]
    qs.sort(key=lambda q: q["qid"])
    order = np.random.default_rng([C.SEEDS["order"], C.CODE, 0 if stage == "dev" else 1]).permutation(len(qs))
    return [dict(qs[i], order=k) for k, i in enumerate(order)]


def yes_no_scorer(P):
    import torch
    unk = P.tok.unk_token_id
    yes = [P.tok.convert_tokens_to_ids(t) for t in C.YES]
    no = [P.tok.convert_tokens_to_ids(t) for t in C.NO]
    assert unk not in yes + no and len(set(yes + no)) == 8, (yes, no)

    def score(logits):
        lp = torch.log_softmax(logits.float(), -1)
        ly, ln = torch.logsumexp(lp[yes], -1), torch.logsumexp(lp[no], -1)
        return ("yes" if ly > ln else "no"), float((ly - ln).abs()), float(ly), float(ln)
    return score, dict(yes=yes, no=no)


def make_runner(P, images, prompt_of, score, gen_check=False):
    """one(q, rng) -> row with the three passes; prompt_of(q) builds the prompt, score(logits) the forced choice."""
    import torch
    from PIL import Image
    from exp.exp_f1_paired import TimedKeepKV, first_ties
    from exp.exp_reuse import Reuse
    R = Reuse(P)
    sync, t = torch.cuda.synchronize, time.perf_counter

    def load(q):
        meta = images[q["image_id"]]
        a = t()
        img = Image.open(ROOT_DIR / meta["path"]).convert("RGB")
        img.load()
        return img, prompt_of(q), (t() - a) * 1e3

    def full(img, pr):
        sync(); a = t()
        inputs = P.full_inputs(img, pr); sync(); b = t()
        logits = P.model(**inputs, use_cache=False).logits[0, -1]; sync(); c = t()
        ans, m, l1, l2 = score(logits); sync(); d = t()
        return dict(full_answer=ans, full_margin=m, full_logp_1=l1, full_logp_2=l2, full_first_ties=first_ties(logits),
                    n_full_tokens=int((inputs["input_ids"] == P.img_tok).sum()),
                    ms_H_pre=(b - a) * 1e3, ms_H_fwd=(c - b) * 1e3, ms_H_score=(d - c) * 1e3)

    def cheap_and_resume(img, pr):
        rec, ends = {}, {}

        def mark(name, at=None):
            if at is None:
                sync()
                at = t()
            ends[name] = at
        real = P.lm
        sync(); a = t()
        pix = P.base_pixel(img); sync(); b = t()
        packed, _ = P.base_encode(pix); sync(); c = t()
        kv = P.lm = TimedKeepKV(real, sync, t)
        try:
            out = P.prefill(packed, pr, use_cache=True)
            d = kv.t_prefill
            ans, m, l1, l2 = score(out.logits[0, -1]); sync(); e = t()
            del out
        finally:
            P.lm = real
        pre, post = P._split_ids(pr)
        rec.update(base_answer=ans, base_margin=m, base_logp_1=l1, base_logp_2=l2, ms_S_pre=(b - a) * 1e3, ms_S_enc=(c - b) * 1e3,
                   ms_P=(d - c) * 1e3, ms_M=(e - d) * 1e3, kv_injected=kv.n_injected, kv_len=kv.cache.get_seq_length(),
                   kv_expected=len(pre) + N_BASE + 1 + len(post), base_first_ties=first_ties(kv.last_logits[0]))
        kv.last_logits = []
        mark("R_start")
        packed_f, t_pre = R.crop_features(img, pr, packed)
        mark("R_pre", t_pre)
        mark("R_vis")
        o2 = R.continue_prefill(kv.cache, len(pre) + N_BASE, packed_f, post)
        mark("R_cont")
        ans_r, m_r, _, _ = score(o2.logits[0, -1])
        mark("R_dec")
        rec.update(resume_answer=ans_r, resume_margin=m_r, resume_first_ties=first_ties(o2.logits[0, -1]),
                   n_resume_tokens=int(packed_f.shape[0]),
                   ms_R_pre=(ends["R_pre"] - ends["R_start"]) * 1e3, ms_R_vis=(ends["R_vis"] - ends["R_pre"]) * 1e3,
                   ms_R_cont=(ends["R_cont"] - ends["R_vis"]) * 1e3, ms_R_dec=(ends["R_dec"] - ends["R_cont"]) * 1e3)
        return rec

    def one(q, rng):
        img, pr, io = load(q)
        meta = images[q["image_id"]]
        rec = dict(qid=q["qid"], order=q.get("order"), image_id=q["image_id"], group=q.get(C.GROUP_KEY, q.get("obs_group")),
                   correct=q.get("correct"), manifest_split=q["split"], width=meta["width"], height=meta["height"], io_ms=round(io, 3))
        first = rng.choice(["cheap", "full"])
        rec["first_path"] = first
        for path in (first, "full" if first == "cheap" else "cheap"):
            rec.update(full(img, pr) if path == "full" else cheap_and_resume(img, pr))
        rec["changed"] = rec["base_answer"] != rec["full_answer"]
        rec["resume_ne_full"] = rec["resume_answer"] != rec["full_answer"]
        if gen_check:  # outside the clock: the full path's free greedy answer
            g = P.full_generate(img, pr, max_new=4)
            word = g["text"].strip().lower().split()[0].strip(".,!:;") if g["text"].strip() else ""
            rec.update(full_gen_text=g["text"], full_gen_word=word, full_gen_agrees=(word == rec["full_answer"]))
        for key in list(rec):
            if key.startswith("ms_"):
                rec[key] = round(rec[key], 3)
        return rec
    return one


def config(P, stage, ids):
    import torch
    import transformers
    from models.llava_next import MAX_NEW, MODEL, REVISION
    return dict(model=MODEL, revision=REVISION, dtype="float16", attn_implementation=P.attn, batch_size=1, dataset="pope",
                stage=stage, device=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__,
                python=platform.python_version(), seeds=C.SEEDS, yes_tokens=list(C.YES), no_tokens=list(C.NO), token_ids=ids,
                max_new_tokens=MAX_NEW,
                timing="per stage, CUDA-synchronised; image decode excluded (io_ms); cheap prefill end marked inside KeepKV",
                resume="crop the cheap KV to prefix + 576 base patches; vision on crops only; forced choice on the continuation")


def out_dir(stage):
    return C.DEV_DIR if stage == "dev" else C.P1_DIR


def run(stage, shard, n_shards):
    import torch
    from exp.exp_f1_paired import load_model
    if stage == "caltest":
        C.load_lock()  # the CAL/TEST runs come after the POPE lock
    out = out_dir(stage)
    out.mkdir(parents=True, exist_ok=True)
    plan = make_plan(stage)
    qids = [q["qid"] for q in plan]
    pj = out / "plan.json"
    if pj.exists() and json.loads(pj.read_text())["qids"] != qids:
        raise SystemExit("plan.json differs from the plan this code builds")
    if not pj.exists():
        pj.write_text(json.dumps(dict(n=len(plan), qids=qids)))
    mine = [q for q in plan if q["order"] % n_shards == shard]
    tp = out / f"traces_shard{shard}.jsonl"
    done = {json.loads(l)["qid"] for l in open(tp)} if tp.exists() else set()
    todo = [q for q in mine if q["qid"] not in done]
    print(f"[{datetime.now():%H:%M:%S}] POPE {stage} shard {shard}/{n_shards}: {len(mine)} questions, done {len(done)}, to run {len(todo)}",
          flush=True)
    if not todo:
        return
    P = load_model()
    score, ids = yes_no_scorer(P)
    imgs = C.images()
    one = make_runner(P, imgs, lambda q: P.prompt_open(q["question"]), score, gen_check=(stage == "dev"))
    check_run_config(out / f"run_config_shard{shard}.json", config(P, stage, ids))
    with torch.inference_mode():
        for q in todo[:3]:  # warm-up, not recorded
            one(q, random.Random(0))
        t0, bad = time.time(), 0
        with open(tp, "a") as f:
            for k, q in enumerate(todo):
                rec = one(q, random.Random(f"{C.SEEDS['path_order']}:{q['qid']}"))
                rec["dev_band"] = q.get("dev_band")
                bad += rec["kv_injected"] != 1 or rec["kv_len"] != rec["kv_expected"]
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if (k + 1) % 100 == 0 or k + 1 == len(todo):
                    print(f"[{datetime.now():%H:%M:%S}] {k + 1}/{len(todo)} done, {(time.time() - t0) / (k + 1):.2f} s/question, "
                          f"cache-check failures {bad}", flush=True)


def merge(stage):
    out = out_dir(stage)
    plan = json.loads((out / "plan.json").read_text())["qids"]
    rows = {}
    for p in sorted(out.glob("traces_shard*.jsonl")):
        for r in C.read_jsonl(p):
            if r["qid"] in rows:
                raise SystemExit(f"duplicate qid {r['qid']} across shards")
            rows[r["qid"]] = r
    missing = [q for q in plan if q not in rows]
    if missing or len(rows) != len(plan):
        raise SystemExit(f"{len(missing)} of {len(plan)} questions missing (or extra rows); finish the shards first")
    cfgs = [json.loads(p.read_text()) for p in sorted(out.glob("run_config_shard*.json"))]
    if any(c != cfgs[0] for c in cfgs):
        raise SystemExit("shards ran under different configurations")
    with open(out / "traces.jsonl", "w") as f:
        for q in plan:
            f.write(json.dumps(rows[q]) + "\n")
    health = dict(n=len(plan), shards=len(cfgs),
                  kv_check_failures=sum(r["kv_injected"] != 1 or r["kv_len"] != r["kv_expected"] for r in rows.values()),
                  merged=datetime.now().isoformat(timespec="seconds"))
    (out / "progress.json").write_text(json.dumps(health, indent=1))
    print(json.dumps(health, indent=1))


def regress_sugarcrepe(n):
    """This runner with the A/B scorer and the SugarCrepe prompt must reproduce exp_f1_paired's SugarCrepe passes on the
    first n pilot_dev questions (same GPU, same seeds): answers equal, margins equal to float tolerance."""
    import torch
    from exp.exp_f1_paired import load_model, make_runner as f1_runner
    plan = sorted(C.read_jsonl(E2_DIR / "plan.jsonl"), key=lambda q: q["order"])[:n]
    P = load_model()
    imgs = {x["image_id"]: x for x in C.read_jsonl(MANIFEST_DIR / "sugarcrepe_images.jsonl")}
    _, ref_one = f1_runner(P, "sugarcrepe", imgs)
    mine = make_runner(P, imgs, lambda q: P.prompt(q["option_a"], q["option_b"]), P.score)
    rows = []
    with torch.inference_mode():
        ref_one(plan[0], random.Random(0)); mine(plan[0], random.Random(0))
        for q in plan:
            a, b = ref_one(q, random.Random(0)), mine(q, random.Random(0))
            rows.append(dict(qid=q["qid"], cheap_same=a["base_answer"] == b["base_answer"], full_same=a["full_answer"] == b["full_answer"],
                             resume_same=a["resume_answer"] == b["resume_answer"], margin_diff=abs(a["base_margin"] - b["base_margin"]),
                             full_margin_diff=abs(a["full_margin"] - b["full_margin"]), resume_margin_diff=abs(a["resume_margin"] - b["resume_margin"]),
                             kv_same=a["kv_len"] == b["kv_len"] == b["kv_expected"]))
    summ = {k: (sum(r[k] for r in rows) if isinstance(rows[0][k], bool) else max(r[k] for r in rows)) for k in rows[0] if k != "qid"}
    summ.update(n=len(rows), device=torch.cuda.get_device_name(0), rows=rows)
    C.DEV_DIR.mkdir(parents=True, exist_ok=True)
    (C.DEV_DIR / "regress_sugarcrepe.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1))


def main():
    ap = argparse.ArgumentParser(description="POPE paired native / cheap / forced-resume runs")
    ap.add_argument("--stage", choices=["dev", "caltest"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--regress_sc", type=int, default=0)
    a = ap.parse_args()
    if a.regress_sc:
        regress_sugarcrepe(a.regress_sc)
    elif a.merge:
        merge(a.stage)
    else:
        run(a.stage, a.shard, a.n_shards)


if __name__ == "__main__":
    main()
