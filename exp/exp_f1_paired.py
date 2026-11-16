"""F1: paired native / cheap / forced-resume runs on every non-DEV question (CAL and TEST of all five rounds), once.

    python -m exp.exp_f1_paired --dataset sugarcrepe --check_dev 30          # GPU: reproduce 30 pilot_dev traces first
    python -m exp.exp_f1_paired --dataset sugarcrepe [--shard 0 --n_shards 2] # GPU: the paired runs (resume)
    python -m exp.exp_f1_paired --dataset sugarcrepe --merge                  # CPU: merge shards, write counts only
    (or bash scripts/f1_paired.sh)

What runs:
  full    the native reference, as E2 / E5: SugarCrepe processor + model forward + A/B score (stages ms_H_pre, ms_H_fwd,
          ms_H_score); TextVQA NextPaths.full_generate (ms_H). Also the number of tied maxima at the first position.
  cheap   the cheap path of the DEV runs, called unchanged: NextPaths.base_pixel, base_encode, prefill (+ score, SugarCrepe) and
          exp_signals.decode_with_signals (TextVQA), with P.lm wrapped by KeepKV (exp/timing_tools.py) so the cheap prefill's
          DynamicCache is kept. Stages ms_S_pre, ms_S_enc, ms_P (prefill) and ms_M (score, SugarCrepe) or ms_D (decode
          after the prefill, TextVQA; the wrapper synchronises once at the end of the prefill call to split the two).
          TextVQA also records the five confidence signals and the margin of every step, recomputed outside the clock from
          the logits the wrapper keeps; their minimum must equal seq_min_margin bit for bit.
  resume  a forced fallback from that cache: exp_reuse.Reuse.crop_features and continue_prefill (crop the cache to the
          text prefix + 576 base-view patches, vision tower on the crops only, prefill crops + question), then the A/B
          score (SugarCrepe) or draft verification (TextVQA). The first token of the continuation is the argmax of the
          logits (first maximum), the rule native generate uses, not exp_reuse's topk; tied maxima
          are counted. Stages ms_R_pre, ms_R_vis, ms_R_cont, ms_R_dec.
  The full and the cheap + resume passes run in a random order per question (seeded), image decoding excluded (io_ms).
The resume answer r_i does not depend on the threshold (the draft is the whole cheap answer), so one forced resume per
question gives the loss of every candidate.

Blind run. F1 runs once over all non-DEV questions; the outputs are only written to disk. Neither the
log nor the merge step prints anything computed from model outputs (only counts and run health: questions done, time,
cache checks). Calibration reads only each round's CAL sample (exp_f2_certify.py); TEST metrics are computed only after
every certificate is written (exp_f2_certify.py --evaluate).

--check_dev N runs the same code on the first N pilot_dev questions of E2 / E5 (by run order) and compares answers and
margins with those traces, as a check before the paired runs. It reads only DEV data and writes results/F1_paired_caltest/<dataset>/check_dev.json.

Outputs: results/F1_paired_caltest/<dataset>/: run_config.json, plan.json (the fixed question order),
traces_shard<k>.jsonl (one row per question), traces.jsonl (merged, sorted by plan order), progress.json (counts).
"""
import argparse
import json
import platform
import random
import time
from datetime import datetime

import numpy as np

from root import E2_DIR, E5_DIR, MANIFEST_DIR, RESULTS_DIR
from utils.tools import check_run_config, read_jsonl

OUT_ROOT = RESULTS_DIR / "F1_paired_caltest"
ORDER_SEED, PATH_SEED = 20261104, 20261105
N_BASE = 576
GROUP_KEY = {"sugarcrepe": "obs_group", "textvqa": "qtype"}
TRANSFORMERS_VERSION = "4.47.0"


def make_plan(dataset):
    """All non-DEV questions in a seeded order (answer-blind), with the manifest split for reference."""
    qs = [q for q in read_jsonl(MANIFEST_DIR / f"{dataset}_questions.jsonl") if q["split"] != "DEV"]
    qs.sort(key=lambda q: q["qid"])
    order = np.random.default_rng([ORDER_SEED, 1 if dataset == "sugarcrepe" else 2]).permutation(len(qs))
    return [dict(qs[i], order=k) for k, i in enumerate(order)]


# ------------------------------------------------------------------------------------------------ the three passes
class TimedKeepKV:
    """KeepKV (exp/timing_tools.py) plus: one synchronised time mark at the end of the cache-creating call (the cheap prefill),
    and a copy of the last-position logits of every call (for the per-step margins, computed outside the clock)."""

    def __init__(self, lm, sync, clock):
        from transformers import DynamicCache
        self.lm, self.new_cache, self.cache, self.n_injected = lm, DynamicCache, None, 0
        self.sync, self.clock, self.t_prefill, self.last_logits = sync, clock, None, []

    def __call__(self, *args, **kwargs):
        inject = kwargs.get("use_cache") and kwargs.get("past_key_values") is None
        if inject:
            kwargs["past_key_values"] = self.cache = self.new_cache()
            self.n_injected += 1
        out = self.lm(*args, **kwargs)
        self.last_logits.append(out.logits[0, -1].detach().clone())
        if inject:
            self.sync()
            self.t_prefill = self.clock()
        return out

    def __getattr__(self, name):
        return getattr(self.lm, name)


def first_ties(logits):
    v = logits.float()
    return int((v == v.max()).sum())


def greedy_argmax(P, first_logits, cache, max_new):
    """Greedy decoding from logits at the first position: argmax (first maximum) at every step, as native generate."""
    from models.llava_next import EOS_ID
    import torch
    toks = [int(first_logits.argmax())]
    while len(toks) < max_new and toks[-1] != EOS_ID:
        o = P.lm(inputs_embeds=P.emb(torch.tensor([[toks[-1]]], device=P.dev)), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        toks.append(int(o.logits[0, -1].argmax()))
    return toks


def verify_draft_argmax(P, first_logits, cache, draft, max_new):
    """exp_reuse.Reuse.verify_draft with the native tie rule: the first token is the argmax of the logits. One forward
    pass checks every draft token against the full model's greedy choice; keep the longest matching prefix, replace the
    first mismatch by the full model's token and continue greedily. Returns (tokens, number of draft tokens kept)."""
    from models.llava_next import EOS_ID
    import torch
    n0 = cache.get_seq_length()
    pred0 = int(first_logits.argmax())
    if not draft or pred0 != draft[0]:
        return greedy_argmax(P, first_logits, cache, max_new), 0
    k = len(draft)
    preds = [pred0]
    if k > 1:
        o = P.lm(inputs_embeds=P.emb(torch.tensor([draft[:-1]], device=P.dev)), past_key_values=cache, use_cache=True)
        preds += [int(x) for x in o.logits[0].argmax(-1).tolist()]
        cache = o.past_key_values
    j = next((i for i in range(k) if preds[i] != draft[i]), k)
    if j == k and (draft[-1] == EOS_ID or k >= max_new):
        return list(draft), k
    cache.crop(n0 + j)
    toks = list(draft[:j]) + ([preds[j]] if j < k else [])
    while len(toks) < max_new and toks[-1] != EOS_ID:
        o = P.lm(inputs_embeds=P.emb(torch.tensor([[toks[-1]]], device=P.dev)), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        toks.append(int(o.logits[0, -1].argmax()))
    return toks, j


def resume_fallback(P, R, dataset, img, prompt, cache, packed, draft, marks=None):
    """The resume continuation from the kept cheap cache. marks(name) records stage ends (optional).
    Returns (answer: option letter or raw text, info)."""
    from models.llava_next import MAX_NEW
    pre, post = P._split_ids(prompt)
    packed_f, t_pre = R.crop_features(img, prompt, packed)  # synchronises once, after the processor
    if marks:
        marks("R_pre", t_pre)
        marks("R_vis")
    o2 = R.continue_prefill(cache, len(pre) + N_BASE, packed_f, post)
    if marks:
        marks("R_cont")
    info = dict(n_full_tokens=int(packed_f.shape[0]), resume_first_ties=first_ties(o2.logits[0, -1]))
    if dataset == "sugarcrepe":
        ans, m, la, lb = P.score(o2.logits[0, -1])
        info.update(resume_margin=m)
    else:
        toks, kept = verify_draft_argmax(P, o2.logits[0, -1], o2.past_key_values, draft, MAX_NEW)
        ans = P.decode_text(toks)
        info.update(draft_kept=kept, resume_n_new=len(toks))
    if marks:
        marks("R_dec")
    return ans, info


def make_runner(P, dataset, images):
    """load(question) -> (img, prompt, io_ms, extra); one_question(q) -> trace row (all three passes)."""
    import torch
    from exp.exp_reuse import Reuse
    from exp.exp_signals import decode_with_signals
    from utils.vqa_eval import accuracy, normalize
    R = Reuse(P)
    sync, t = torch.cuda.synchronize, time.perf_counter

    if dataset == "sugarcrepe":
        from exp.sugarcrepe_inputs import prepare

        def load(q):
            it = prepare(P, q, images)
            return it["img"], it["prompt"], it["io_ms"], dict(category=q["category"], correct=q["correct"],
                                                               grid=it["grid"], aspect=it["aspect"])
    else:
        from exp.exp_paired_textvqa import load as load_question

        def load(q):
            img, pr, io = load_question(P, q, images)
            return img, pr, io, dict(question=q["question"])

    def full(q, img, pr):
        if dataset == "sugarcrepe":
            sync(); a = t()
            inputs = P.full_inputs(img, pr); sync(); b = t()
            logits = P.model(**inputs, use_cache=False).logits[0, -1]; sync(); c = t()
            ans, m, la, lb = P.score(logits); sync(); d = t()
            return dict(full_answer=ans, full_margin=m, full_logp_a=la, full_logp_b=lb, full_first_ties=first_ties(logits),
                        n_full_tokens=int((inputs["input_ids"] == P.img_tok).sum()),
                        ms_H_pre=(b - a) * 1e3, ms_H_fwd=(c - b) * 1e3, ms_H_score=(d - c) * 1e3)
        sync(); a = t()
        res = P.full_generate(img, pr); sync(); b = t()
        return dict(full_text=res["text"], full_norm=normalize(res["text"]), full_tokens=res["tokens"], full_margin=res["margin"],
                    full_n_new=res["n_new"], full_truncated=res["truncated"], full_acc=accuracy(res["text"], q["answers"]),
                    n_full_tokens=res["n_visual_tokens"], ms_H=(b - a) * 1e3)

    def cheap_and_resume(q, img, pr):
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
            if dataset == "sugarcrepe":
                out = P.prefill(packed, pr, use_cache=True)
                d = kv.t_prefill
                ans, m, la, lb = P.score(out.logits[0, -1]); sync(); e = t()
                del out
                rec.update(base_answer=ans, base_margin=m, base_logp_a=la, base_logp_b=lb, ms_P=(d - c) * 1e3, ms_M=(e - d) * 1e3)
                draft = None
            else:
                text, toks, sig = decode_with_signals(P, packed, pr); sync(); e = t()
                d = kv.t_prefill
                rec.update(base_text=text, base_norm=normalize(text), base_tokens=toks, base_n_new=len(toks),
                           base_acc=accuracy(text, q["answers"]), ms_P=(d - c) * 1e3, ms_D=(e - d) * 1e3, **sig)
                draft = toks
        finally:
            P.lm = real
        pre, post = P._split_ids(pr)
        n_prompt = len(pre) + N_BASE + 1 + len(post)
        rec.update(ms_S_pre=(b - a) * 1e3, ms_S_enc=(c - b) * 1e3, kv_injected=kv.n_injected,
                   kv_len=kv.cache.get_seq_length(), kv_expected=n_prompt + (0 if draft is None else len(draft) - 1),
                   base_first_ties=first_ties(kv.last_logits[0]))
        if draft is not None:  # per-step margins from the kept logits (outside the clock)
            steps = []
            for lg in kv.last_logits[:len(draft)]:
                top = torch.topk(torch.softmax(lg.float(), -1), 2).values
                steps.append(float(top[0] - top[1]))
            rec.update(step_margins=steps, step_min_eq_signal=min(steps) == rec["seq_min_margin"])
        kv.last_logits = []
        # forced resume (fallback) from the kept cache
        mark("R_start")
        ans_r, info = resume_fallback(P, R, dataset, img, pr, kv.cache, packed, draft, marks=mark)
        rec.update(info)
        rec.update(ms_R_pre=(ends["R_pre"] - ends["R_start"]) * 1e3, ms_R_vis=(ends["R_vis"] - ends["R_pre"]) * 1e3,
                   ms_R_cont=(ends["R_cont"] - ends["R_vis"]) * 1e3, ms_R_dec=(ends["R_dec"] - ends["R_cont"]) * 1e3)
        if dataset == "sugarcrepe":
            rec.update(resume_answer=ans_r)
        else:
            rec.update(resume_text=ans_r, resume_norm=normalize(ans_r))
        return rec

    def one_question(q, rng):
        img, pr, io, extra = load(q)
        meta = images[q["image_id"]]
        rec = dict(qid=q["qid"], order=q.get("order"), image_id=q["image_id"], group=q[GROUP_KEY[dataset]],
                   manifest_split=q["split"], width=meta["width"], height=meta["height"], io_ms=round(io, 3), **extra)
        first = rng.choice(["cheap", "full"])
        rec["first_path"] = first
        for path in (first, "full" if first == "cheap" else "cheap"):
            rec.update(full(q, img, pr) if path == "full" else cheap_and_resume(q, img, pr))
        if dataset == "sugarcrepe":
            rec["changed"] = rec["base_answer"] != rec["full_answer"]
            rec["resume_ne_full"] = rec["resume_answer"] != rec["full_answer"]
        else:
            rec["changed"] = rec["base_norm"] != rec["full_norm"]
            rec["resume_ne_full"] = rec["resume_norm"] != rec["full_norm"]
        for key in list(rec):
            if key.startswith("ms_"):
                rec[key] = round(rec[key], 3)
        return rec

    return load, one_question


def load_model():
    import torch
    import transformers
    from models.llava_next import NextPaths
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise SystemExit(f"transformers {transformers.__version__}: this code was checked with {TRANSFORMERS_VERSION}")
    P = NextPaths("cuda:0")
    torch.cuda.synchronize()
    return P


def config(P, dataset):
    import torch
    import transformers
    from models.llava_next import MAX_NEW, MODEL, REVISION
    return dict(model=MODEL, revision=REVISION, dtype="float16", attn_implementation=P.attn, batch_size=1, greedy=True,
                max_new_tokens=MAX_NEW, dataset=dataset, device=torch.cuda.get_device_name(0), torch=torch.__version__,
                transformers=transformers.__version__, python=platform.python_version(), order_seed=ORDER_SEED,
                path_seed=PATH_SEED,
                timing="per stage, CUDA-synchronised; image decode excluded (io_ms); cheap prefill end marked by one sync "
                       "inside the KeepKV wrapper",
                resume="crop the cheap KV to prefix + 576 base patches; vision on crops only; first continuation token = "
                       "argmax (first maximum); TextVQA verifies the cheap answer as a draft")


# ------------------------------------------------------------------------------------------------ entry points
def run(dataset, shard, n_shards):
    import torch
    out = OUT_ROOT / dataset
    out.mkdir(parents=True, exist_ok=True)
    plan = make_plan(dataset)
    pj = out / "plan.json"
    if pj.exists() and json.loads(pj.read_text())["qids"] != [q["qid"] for q in plan]:
        raise SystemExit("plan.json differs from the plan this code builds")
    if not pj.exists():
        pj.write_text(json.dumps(dict(n=len(plan), order_seed=ORDER_SEED, qids=[q["qid"] for q in plan])))
    mine = [q for q in plan if q["order"] % n_shards == shard]
    tp = out / f"traces_shard{shard}.jsonl"
    done = {json.loads(l)["qid"] for l in open(tp)} if tp.exists() else set()
    todo = [q for q in mine if q["qid"] not in done]
    print(f"[{datetime.now():%H:%M:%S}] F1 {dataset} shard {shard}/{n_shards}: {len(mine)} questions, done {len(done)}, "
          f"to run {len(todo)}", flush=True)
    if not todo:
        return
    P = load_model()
    check_run_config(out / f"run_config_shard{shard}.json", config(P, dataset))
    images = {x["image_id"]: x for x in read_jsonl(MANIFEST_DIR / f"{dataset}_images.jsonl")}
    load, one = make_runner(P, dataset, images)
    with torch.inference_mode():
        for q in todo[:3]:  # warm-up, not recorded
            one(q, random.Random(0))
        t0, bad_kv = time.time(), 0
        with open(tp, "a") as f:
            for k, q in enumerate(todo):
                rec = one(q, random.Random(f"{PATH_SEED}:{q['qid']}"))
                bad_kv += rec["kv_injected"] != 1 or rec["kv_len"] != rec["kv_expected"]
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if (k + 1) % 100 == 0 or k + 1 == len(todo):
                    el = time.time() - t0
                    print(f"[{datetime.now():%H:%M:%S}] {k + 1}/{len(todo)} done, {el / (k + 1):.2f} s/question, "
                          f"cache-check failures {bad_kv}", flush=True)


def merge(dataset):
    """Merge the shards (sorted by plan order); write counts only."""
    out = OUT_ROOT / dataset
    plan = json.loads((out / "plan.json").read_text())["qids"]
    rows = {}
    for p in sorted(out.glob("traces_shard*.jsonl")):
        for r in read_jsonl(p):
            rows[r["qid"]] = r
    missing = [q for q in plan if q not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(plan)} questions missing; run the shards to completion first")
    with open(out / "traces.jsonl", "w") as f:
        for q in plan:
            f.write(json.dumps(rows[q]) + "\n")
    cfgs = [json.loads(p.read_text()) for p in sorted(out.glob("run_config_shard*.json"))]
    same = all({k: v for k, v in c.items()} == cfgs[0] for c in cfgs)
    health = dict(n=len(plan), shards=len(cfgs), configs_identical=same,
                  kv_check_failures=sum(r["kv_injected"] != 1 or r["kv_len"] != r["kv_expected"] for r in rows.values()),
                  step_min_mismatch=sum(r.get("step_min_eq_signal") is False for r in rows.values()),
                  merged=datetime.now().isoformat(timespec="seconds"))
    (out / "progress.json").write_text(json.dumps(health, indent=1))
    print(json.dumps(health, indent=1))


def check_dev(dataset, n):
    """The F1 code on the first n pilot_dev questions of E2 / E5: answers and margins against the DEV traces."""
    import torch
    src = E2_DIR if dataset == "sugarcrepe" else E5_DIR
    traces = {r["qid"]: r for r in read_jsonl(src / "traces.jsonl")}
    sig = {r["qid"]: r for r in read_jsonl(RESULTS_DIR / "E8_textvqa_signals_dev" / "signals.jsonl")} if dataset == "textvqa" else {}
    plan = sorted(read_jsonl(src / "plan.jsonl"), key=lambda q: q["order"])[:n]
    P = load_model()
    images = {x["image_id"]: x for x in read_jsonl(MANIFEST_DIR / f"{dataset}_images.jsonl")}
    load, one = make_runner(P, dataset, images)
    rows = []
    with torch.inference_mode():
        one(plan[0], random.Random(0))
        for q in plan:
            r, ref = one(q, random.Random(0)), traces[q["qid"]]
            if dataset == "sugarcrepe":
                rows.append(dict(qid=q["qid"], cheap_same=r["base_answer"] == ref["base_answer"], full_same=r["full_answer"] == ref["full_answer"],
                                 margin_diff=abs(r["base_margin"] - ref["base_margin"]), full_margin_diff=abs(r["full_margin"] - ref["full_margin"]),
                                 resume_eq_full=r["resume_answer"] == r["full_answer"], kv_ok=r["kv_len"] == r["kv_expected"],
                                 ms_H=r["ms_H_pre"] + r["ms_H_fwd"] + r["ms_H_score"], ms_H_dev=ref["ms_H_pre"] + ref["ms_H_fwd"] + ref["ms_H_score"]))
            else:
                rows.append(dict(qid=q["qid"], cheap_same=r["base_text"] == ref["base_text"], full_same=r["full_norm"] == ref["full_norm"],
                                 margin_diff=abs(r["seq_min_margin"] - sig[q["qid"]]["seq_min_margin"]),
                                 full_margin_diff=abs(r["full_margin"] - ref["full_margin"]), step_min_ok=r["step_min_eq_signal"],
                                 resume_eq_full=r["resume_norm"] == r["full_norm"], kv_ok=r["kv_len"] == r["kv_expected"],
                                 ms_H=r["ms_H"], ms_H_dev=ref["ms_H"]))
    summ = dict(n=len(rows), device=torch.cuda.get_device_name(0),
                cheap_same=sum(x["cheap_same"] for x in rows), full_same=sum(x["full_same"] for x in rows),
                max_margin_diff=max(x["margin_diff"] for x in rows), max_full_margin_diff=max(x["full_margin_diff"] for x in rows),
                resume_eq_full=sum(x["resume_eq_full"] for x in rows), kv_ok=sum(x["kv_ok"] for x in rows),
                mean_ms_H=float(np.mean([x["ms_H"] for x in rows])), mean_ms_H_dev=float(np.mean([x["ms_H_dev"] for x in rows])),
                rows=rows)
    out = OUT_ROOT / dataset
    out.mkdir(parents=True, exist_ok=True)
    (out / "check_dev.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1))


def main():
    ap = argparse.ArgumentParser(description="F1: paired native / cheap / forced-resume runs on CAL + TEST")
    ap.add_argument("--dataset", required=True, choices=list(GROUP_KEY))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--check_dev", type=int, default=0)
    a = ap.parse_args()
    if a.merge:
        merge(a.dataset)
    elif a.check_dev:
        check_dev(a.dataset, a.check_dev)
    else:
        run(a.dataset, a.shard, a.n_shards)


if __name__ == "__main__":
    main()
