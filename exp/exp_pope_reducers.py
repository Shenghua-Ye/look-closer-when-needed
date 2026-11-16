"""POPE: fixed-budget token reducers (2x2 pooling, VisionZip 640) on every round-0 TEST question, as exp_f4_reducers does
for the other two datasets (no guarantee; their own budgets). New file; run after the POPE certificates exist.

    python -m exp.exp_pope_reducers --reducer pool        # GPU (resumes)
    python -m exp.exp_pope_reducers --reducer visionzip
    python -m exp.exp_pope_reducers --evaluate            # CPU

The reduced path is exp_pooled's code, unchanged; the answer is the forced yes/no choice of exp_pope_paired. The reference
answer and the full-pass time are P1's native full pass on the same question.
Outputs: results/P4_pope_reducers_test/pope_<reducer>/traces.jsonl, results/P4_pope_reducers_test/summary.{md,json}.
"""
import argparse
import json
import platform
import time

import numpy as np

from exp import pope_common as C
from root import RESULTS_DIR, ROOT_DIR
from utils.tools import check_run_config

OUT = RESULTS_DIR / "P4_pope_reducers_test"


def round0_test():
    test = set(json.loads((C.LOCK_DIR / "splits_pope.json").read_text())["rounds"][0]["TEST"])
    return sorted((q for q in C.questions() if q["image_id"] in test), key=lambda q: q["qid"])


def run(reducer):
    import torch
    import transformers
    from PIL import Image
    from exp.exp_f1_paired import load_model
    from exp.exp_pooled import CHEAP_PATHS, REDUCERS, check_packing, image_inputs
    from exp.exp_pope_paired import yes_no_scorer
    from utils.tools import sync
    out = OUT / f"pope_{reducer}"
    out.mkdir(parents=True, exist_ok=True)
    qs = round0_test()
    tp = out / "traces.jsonl"
    done = {json.loads(l)["qid"] for l in open(tp)} if tp.exists() else set()
    todo = [q for q in qs if q["qid"] not in done]
    print(f"POPE {reducer}: {len(qs)} round-0 TEST questions, done {len(done)}, to run {len(todo)}", flush=True)
    if not todo:
        return
    P = load_model()
    score, ids = yes_no_scorer(P)
    check_run_config(out / "run_config.json", dict(dataset="pope", reducer=reducer, cheap_path=CHEAP_PATHS[reducer], token_ids=ids,
                                                   device=torch.cuda.get_device_name(0), torch=torch.__version__,
                                                   transformers=transformers.__version__, python=platform.python_version()))
    images = C.images()

    def load(q):
        img = Image.open(ROOT_DIR / images[q["image_id"]]["path"]).convert("RGB")
        img.load()
        return img, P.prompt_open(q["question"])
    pack = REDUCERS[reducer]
    t = time.perf_counter
    with torch.inference_mode():
        if reducer == "pool":
            check_packing(P, [load(q) for q in todo[:3]])
        for q in todo[:3]:  # warm-up
            img, pr = load(q)
            pv, sizes = image_inputs(P, img)
            packed, _ = pack(P, pv, sizes)
            score(P.prefill(packed, pr).logits[0, -1])
        with open(tp, "a") as f:
            for k, q in enumerate(todo):
                img, pr = load(q)
                sync(); a = t()
                pv, sizes = image_inputs(P, img); sync(); b = t()
                packed, _ = pack(P, pv, sizes); sync(); c = t()
                ans, m, _, _ = score(P.prefill(packed, pr).logits[0, -1]); sync(); e = t()
                rec = dict(qid=q["qid"], image_id=q["image_id"], n_tokens=int(packed.shape[0]), answer=ans, margin=m,
                           acc=float(ans == q["correct"]), ms_S_pre=(b - a) * 1e3, ms_S_enc=(c - b) * 1e3, ms_rest=(e - c) * 1e3)
                rec["ms_total"] = rec["ms_S_pre"] + rec["ms_S_enc"] + rec["ms_rest"]
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if (k + 1) % 200 == 0:
                    print(f"{k + 1}/{len(todo)}", flush=True)


def evaluate():
    from exp.exp_f2_certify import cp
    cert = C.P2_DIR / "certificates_pope.json"
    if not cert.exists():
        raise SystemExit("TEST metrics only after the POPE certificates are written")
    p1 = {r["qid"]: r for r in C.read_jsonl(C.P1_DIR / "traces.jsonl")}
    sample = json.loads((C.LOCK_DIR / "test_sample_pope.json").read_text())[0]
    res, md = {}, ["# POPE token reducers on round-0 TEST (no guarantee, own budgets)", "",
                   "| reducer | tokens (mean) | saving vs native full | worst group | per-group k/n | R_I | accuracy (full) |", "|---|---|---|---|---|---|---|"]
    H = lambda q: p1[q]["ms_H_pre"] + p1[q]["ms_H_fwd"] + p1[q]["ms_H_score"]
    for red in ("pool", "visionzip"):
        p = OUT / f"pope_{red}" / "traces.jsonl"
        if not p.exists():
            continue
        rows = {r["qid"]: r for r in C.read_jsonl(p)}
        qids = list(rows)
        saving = 1 - sum(rows[q]["ms_total"] for q in qids) / sum(H(q) for q in qids)
        groups = {}
        for x in sample["per_group"]:
            groups.setdefault(x["group"], []).append(rows[x["qid"]]["answer"] != p1[x["qid"]]["full_answer"])
        gr = {g: dict(n=len(v), k=int(sum(v)), rate=float(np.mean(v)), ci=cp(int(sum(v)), len(v))) for g, v in groups.items()}
        ri = [rows[x["qid"]]["answer"] != p1[x["qid"]]["full_answer"] for x in sample["per_image"]]
        acc = float(np.mean([rows[q]["acc"] for q in qids]))
        accf = float(np.mean([p1[q]["full_answer"] == p1[q]["correct"] for q in qids]))
        worst = max(gr, key=lambda g: gr[g]["rate"])
        res[red] = dict(n=len(qids), tokens=float(np.mean([r["n_tokens"] for r in rows.values()])), saving=saving, groups=gr,
                        worst=worst, above_alpha=[g for g, v in gr.items() if v["rate"] > C.ALPHA_MAIN], RI=float(np.mean(ri)),
                        accuracy=acc, accuracy_full=accf)
        md.append(f"| {red} | {res[red]['tokens']:.0f} | {saving:.1%} | {worst} {gr[worst]['rate']:.1%} | "
                  + ", ".join(f"{g} {v['k']}/{v['n']}" for g, v in gr.items()) + f" | {np.mean(ri):.1%} | {acc:.1%} ({accf:.1%}) |")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reducer", choices=["pool", "visionzip"])
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    if a.evaluate:
        evaluate()
    else:
        run(a.reducer)


if __name__ == "__main__":
    main()
