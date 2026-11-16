"""Human check of changed answers (DEV TextVQA): is a "change" (cheap answer != full answer after VQA normalisation) a
material change, or an artefact of string matching? Two annotators label the same seeded sample independently.

    python -m exp.exp_human_eval --build      # writes report/human_eval/: sheet.html, labels_A.csv, labels_B.csv
    python -m exp.exp_human_eval --analyze    # after both CSVs are filled: agreement (Cohen's kappa) and rates

Sample: 150 of the 509 pilot_dev TextVQA questions whose cheap and full answers differ (E5 traces), drawn with seed
20261108; only DEV data is shown. Each annotator sees the image, the question, the two answers in a random order
(labelled X and Y, seeded per item, so the annotator does not know which is the cheap answer) and the ten human answers.
Labels (one per item):
  same      X and Y mean the same thing (e.g. spelling, formatting, abbreviation, extra word with no new content)
  different X and Y give different information
  unsure    cannot tell from the image and answers
Optional second column `better`: X, Y, both, neither (which answer is correct), for an accuracy view.
"""
import argparse
import csv
import json
import random

import numpy as np

from root import E5_DIR, MANIFEST_DIR, ROOT_DIR
from utils.tools import read_jsonl

OUT = ROOT_DIR / "report" / "human_eval"
SEED, N = 20261108, 150
LABELS = ("same", "different", "unsure")


def sample():
    rows = [r for r in read_jsonl(E5_DIR / "traces.jsonl") if r["changed"]]
    rows.sort(key=lambda r: r["qid"])
    pick = np.random.default_rng(SEED).choice(len(rows), N, replace=False)
    return [rows[i] for i in sorted(pick)]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    qs = {q["qid"]: q for q in read_jsonl(MANIFEST_DIR / "textvqa_questions.jsonl")}
    imgs = {x["image_id"]: x for x in read_jsonl(MANIFEST_DIR / "textvqa_images.jsonl")}
    items, key = [], []
    for k, r in enumerate(sample()):
        flip = random.Random(f"{SEED}:{r['qid']}").random() < 0.5
        x, y = (r["full_text"], r["base_text"]) if flip else (r["base_text"], r["full_text"])
        items.append(dict(item=k + 1, qid=r["qid"], question=r["question"], X=x, Y=y, img="../../" + imgs[r["image_id"]]["path"],
                          humans=", ".join(sorted(set(qs[r["qid"]]["answers"])))))
        key.append(dict(item=k + 1, qid=r["qid"], X="full" if flip else "cheap", Y="cheap" if flip else "full"))
    cards = "\n".join(
        f"<div class=c><h3>#{it['item']} &nbsp;<small>{it['qid']}</small></h3><img src='{it['img']}'><p><b>Q:</b> {it['question']}</p>"
        f"<p><b>X:</b> <code>{it['X']}</code><br><b>Y:</b> <code>{it['Y']}</code></p><p class=h><b>Human answers:</b> {it['humans']}</p></div>"
        for it in items)
    (OUT / "sheet.html").write_text(
        "<!DOCTYPE html><html><head><meta charset=utf-8><title>Changed-answer check</title><style>body{font-family:sans-serif;"
        "max-width:900px;margin:auto}.c{border-bottom:1px solid #ccc;padding:12px 0}img{max-width:100%;max-height:420px}"
        "code{font-size:1.1em;background:#f3f3f3;padding:1px 4px}.h{color:#555;font-size:.9em}</style></head><body>"
        "<h1>Changed-answer check (150 items)</h1><p>For each item, enter in your own CSV (labels_A.csv or labels_B.csv), "
        "without discussing: <b>label</b> = same (X and Y mean the same thing: spelling, formatting, abbreviation, an extra "
        "word with no new content), different (they give different information), or unsure. Optional <b>better</b> = X, Y, "
        "both or neither: which answer is correct for the image. Work alone.</p>" + cards + "</body></html>")
    for who in ("A", "B"):
        p = OUT / f"labels_{who}.csv"
        if not p.exists():
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["item", "qid", "label", "better", "note"])
                for it in items:
                    w.writerow([it["item"], it["qid"], "", "", ""])
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    print(f"wrote {OUT}/sheet.html, labels_A.csv, labels_B.csv, key.json ({len(items)} items)")


def analyze():
    from sklearn.metrics import cohen_kappa_score
    lab = {}
    for who in ("A", "B"):
        with open(OUT / f"labels_{who}.csv") as f:
            lab[who] = {int(r["item"]): r for r in csv.DictReader(f)}
    items = sorted(i for i in lab["A"] if lab["A"][i]["label"].strip() and lab["B"].get(i, {}).get("label", "").strip())
    a = [lab["A"][i]["label"].strip().lower() for i in items]
    b = [lab["B"][i]["label"].strip().lower() for i in items]
    bad = [x for x in a + b if x not in LABELS]
    if bad:
        raise SystemExit(f"unknown labels: {sorted(set(bad))}; use {LABELS}")
    kappa = cohen_kappa_score(a, b)
    agree = [x for x, y in zip(a, b) if x == y]
    res = dict(n=len(items), kappa=float(kappa), raw_agreement=float(np.mean([x == y for x, y in zip(a, b)])),
               A={l: a.count(l) for l in LABELS}, B={l: b.count(l) for l in LABELS},
               agreed={l: agree.count(l) for l in LABELS},
               material_rate_agreed=float(agree.count("different") / max(1, len(agree))))
    key = {k["item"]: k for k in json.loads((OUT / "key.json").read_text())}
    for who in ("A", "B"):
        bet = [lab[who][i]["better"].strip().upper() for i in items if lab[who][i]["better"].strip()]
        if bet:
            ch = sum(1 for i in items if lab[who][i]["better"].strip().upper() in ("X", "Y") and key[i][lab[who][i]["better"].strip().upper()] == "cheap")
            fu = sum(1 for i in items if lab[who][i]["better"].strip().upper() in ("X", "Y") and key[i][lab[who][i]["better"].strip().upper()] == "full")
            res[f"better_{who}"] = dict(cheap=ch, full=fu, both=bet.count("BOTH"), neither=bet.count("NEITHER"))
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    x = ap.parse_args()
    build() if x.build else analyze()
