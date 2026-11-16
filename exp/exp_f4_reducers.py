"""F4: fixed-budget token reducers (2x2 pooling, VisionZip 640) on every round-0 TEST question: the external baselines
on TEST (no guarantee; their own budgets).

    python -m exp.exp_f4_reducers --dataset sugarcrepe --reducer pool         # GPU (resumes)
    python -m exp.exp_f4_reducers --dataset textvqa --reducer visionzip
    python -m exp.exp_f4_reducers --evaluate                                  # CPU, after F2 --certify (TEST metrics)

The reduced path is exp_pooled's code, unchanged (pooled_pack / visionzip_pack, image_inputs; the pooling packing check
runs first). It is timed per stage (ms_S_pre = processor on all crops, ms_S_enc = tower + reduction + packing,
then prefill / score or greedy decode). The reference answer and the full-pass time are F1's native full pass on the same
question; only the reduced answers are produced here, so this run reveals nothing about the policy.
Evaluation: per-group change rate on the fixed round-0 per_group TEST sample (k/n, Clopper-Pearson), worst group, the
per-image rate R_I, accuracy, and the replayed saving over all round-0 TEST questions (1 - sum reduced / sum full).
Outputs: results/F4_reducers_test/<dataset>_<reducer>/traces.jsonl, results/F4_reducers_test/summary.{md,json}.
"""
import argparse
import json
import platform
import time

import numpy as np

from root import MANIFEST_DIR, RESULTS_DIR
from utils.tools import check_run_config, read_jsonl

OUT = RESULTS_DIR / "F4_reducers_test"
F1 = RESULTS_DIR / "F1_paired_caltest"
LOCK_DIR = RESULTS_DIR / "F0_protocol_lock"


def round0_test_questions(ds):
    test = set(json.loads((LOCK_DIR / f"splits_{ds}.json").read_text())["rounds"][0]["TEST"])
    return sorted((q for q in read_jsonl(MANIFEST_DIR / f"{ds}_questions.jsonl") if q["image_id"] in test), key=lambda q: q["qid"])


def run(ds, reducer):
    import torch
    import transformers
    from exp.exp_f1_paired import load_model
    from exp.exp_pooled import CHEAP_PATHS, REDUCERS, check_packing, image_inputs
    from utils.tools import sync
    from utils.vqa_eval import accuracy, normalize
    out = OUT / f"{ds}_{reducer}"
    out.mkdir(parents=True, exist_ok=True)
    qs = round0_test_questions(ds)
    tp = out / "traces.jsonl"
    done = {json.loads(l)["qid"] for l in open(tp)} if tp.exists() else set()
    todo = [q for q in qs if q["qid"] not in done]
    print(f"F4 {ds} {reducer}: {len(qs)} round-0 TEST questions, done {len(done)}, to run {len(todo)}", flush=True)
    if not todo:
        return
    P = load_model()
    check_run_config(out / "run_config.json", dict(dataset=ds, reducer=reducer, cheap_path=CHEAP_PATHS[reducer],
                                                   device=torch.cuda.get_device_name(0), torch=torch.__version__,
                                                   transformers=transformers.__version__, python=platform.python_version()))
    images = {x["image_id"]: x for x in read_jsonl(MANIFEST_DIR / f"{ds}_images.jsonl")}
    if ds == "sugarcrepe":
        from exp.sugarcrepe_inputs import prepare

        def load(q):
            it = prepare(P, q, images)
            return it["img"], it["prompt"]
    else:
        from exp.exp_paired_textvqa import load as load_question

        def load(q):
            img, pr, _ = load_question(P, q, images)
            return img, pr
    pack = REDUCERS[reducer]
    t = time.perf_counter
    with torch.inference_mode():
        if reducer == "pool":
            check_packing(P, [load(q) for q in todo[:3]])
        for q in todo[:3]:  # warm-up
            img, pr = load(q)
            pv, sizes = image_inputs(P, img)
            packed, _ = pack(P, pv, sizes)
            P.score(P.prefill(packed, pr).logits[0, -1]) if ds == "sugarcrepe" else P.greedy_from_embeddings(packed, pr)
        with open(tp, "a") as f:
            for k, q in enumerate(todo):
                img, pr = load(q)
                sync(); a = t()
                pv, sizes = image_inputs(P, img); sync(); b = t()
                packed, _ = pack(P, pv, sizes); sync(); c = t()
                rec = dict(qid=q["qid"], image_id=q["image_id"], n_tokens=int(packed.shape[0]), ms_S_pre=(b - a) * 1e3, ms_S_enc=(c - b) * 1e3)
                if ds == "sugarcrepe":
                    logits = P.prefill(packed, pr).logits[0, -1]
                    ans, m, _, _ = P.score(logits); sync(); e = t()
                    rec.update(answer=ans, margin=m, acc=float(ans == q["correct"]), ms_rest=(e - c) * 1e3)
                else:
                    res = P.greedy_from_embeddings(packed, pr); sync(); e = t()
                    rec.update(answer=normalize(res["text"]), text=res["text"], margin=res["margin"],
                               acc=accuracy(res["text"], q["answers"]), ms_rest=(e - c) * 1e3)
                rec["ms_total"] = rec["ms_S_pre"] + rec["ms_S_enc"] + rec["ms_rest"]
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if (k + 1) % 200 == 0:
                    print(f"{k + 1}/{len(todo)}", flush=True)


def evaluate():
    from exp.exp_f2_certify import OUT as F2, cp
    if not (F2 / "certificates.json").exists():
        raise SystemExit("TEST metrics only after the CAL certificates are written (exp_f2_certify --certify)")
    res, md = {}, ["# F4: token reducers on round-0 TEST (no guarantee, own budgets)", "",
                   "| dataset | reducer | tokens (mean) | saving vs native full | worst group | per-group k/n | R_I | accuracy (full) |",
                   "|---|---|---|---|---|---|---|---|"]
    for ds in ("sugarcrepe", "textvqa"):
        f1 = {r["qid"]: r for r in read_jsonl(F1 / ds / "traces.jsonl")}
        sample = json.loads((LOCK_DIR / f"test_sample_{ds}.json").read_text())[0]
        for red in ("pool", "visionzip"):
            p = OUT / f"{ds}_{red}" / "traces.jsonl"
            if not p.exists():
                continue
            rows = {r["qid"]: r for r in read_jsonl(p)}
            full = lambda q: f1[q]["full_answer"] if ds == "sugarcrepe" else f1[q]["full_norm"]
            H = lambda q: (f1[q]["ms_H_pre"] + f1[q]["ms_H_fwd"] + f1[q]["ms_H_score"]) if ds == "sugarcrepe" else f1[q]["ms_H"]
            qids = list(rows)
            saving = 1 - sum(rows[q]["ms_total"] for q in qids) / sum(H(q) for q in qids)
            groups = {}
            for x in sample["per_group"]:
                groups.setdefault(x["group"], []).append(rows[x["qid"]]["answer"] != full(x["qid"]))
            gr = {g: dict(n=len(v), k=int(sum(v)), rate=float(np.mean(v)), ci=cp(int(sum(v)), len(v))) for g, v in groups.items()}
            ri = [rows[x["qid"]]["answer"] != full(x["qid"]) for x in sample["per_image"]]
            acc = float(np.mean([rows[q]["acc"] for q in qids]))
            accf = float(np.mean([f1[q]["full_answer"] == f1[q]["correct"] if ds == "sugarcrepe" else f1[q]["full_acc"] for q in qids]))
            worst = max(gr, key=lambda g: gr[g]["rate"])
            res[f"{ds}_{red}"] = dict(n=len(qids), tokens=float(np.mean([r["n_tokens"] for r in rows.values()])), saving=saving,
                                      groups=gr, worst=worst, RI=float(np.mean(ri)), accuracy=acc, accuracy_full=accf)
            md.append(f"| {ds} | {red} | {res[f'{ds}_{red}']['tokens']:.0f} | {saving:.1%} | {worst} {gr[worst]['rate']:.1%} | "
                      + ", ".join(f"{g} {v['k']}/{v['n']}" for g, v in gr.items()) + f" | {np.mean(ri):.1%} | {acc:.1%} ({accf:.1%}) |")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sugarcrepe", "textvqa"])
    ap.add_argument("--reducer", choices=["pool", "visionzip"])
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    if a.evaluate:
        evaluate()
    else:
        run(a.dataset, a.reducer)


if __name__ == "__main__":
    main()
