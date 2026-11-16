"""E8: confidence signals of the cheap answer on TextVQA pilot DEV. The final study uses seq_min_margin (main) and
first_margin (ablation); exp_frontier.load reads them from here when it builds the candidate thresholds.

    python run.py --task_name signals --dataset textvqa      # GPU; results/E8_textvqa_signals_dev/

Re-runs only the cheap path (base view, greedy decoding, the same code as E5) on the E5 plan and records, per question:
  first_margin       top-1 minus top-2 probability at the first position (E5's margin; checked to match)
  first_p1           top-1 probability at the first position
  first_neg_entropy  minus the entropy of the first-position distribution
  seq_min_margin     the smallest top-1 minus top-2 probability over all generated tokens
  seq_mean_logp      the mean log-probability of the generated tokens
Higher means more confident. The cheap answer must equal E5's (determinism check). No timing is recorded: the cost model
keeps E5's measured stage times, and seq_* signals are read after the whole cheap answer, so their fallback also
wastes the cheap decode.
"""
import json

import numpy as np
import torch

from exp.exp_basic import ExpBasic
from root import E5_DIR, MANIFEST_DIR, RESULTS_DIR
from utils.tools import read_jsonl

OUT = RESULTS_DIR / "E8_textvqa_signals_dev"
SIGNALS = ("first_margin", "first_p1", "first_neg_entropy", "seq_min_margin", "seq_mean_logp")


def _step(logits, first=False):
    """The greedy token exactly as models.llava_next picks it, so ties break the same way: the first token is the top-1
    of the softmax (first_margin), later tokens the argmax of the fp16 logits. Also its probability and the top-1 minus
    top-2 probability."""
    probs = torch.softmax(logits.float(), -1)
    top = torch.topk(probs, 2)
    tok = int(top.indices[0]) if first else int(logits.argmax())
    return probs, tok, float(top.values[0]), float(top.values[0] - top.values[1])


@torch.inference_mode()
def decode_with_signals(P, packed, prompt):
    """Greedy cheap answer from the packed features (the same tokens as NextPaths.greedy_from_embeddings) and its five
    confidence signals. Rules: every generated step counts, including the EOS step and, for an answer truncated at
    MAX_NEW tokens, all MAX_NEW steps; ties break as in the decoder (see _step)."""
    from models.llava_next import EOS_ID, MAX_NEW
    out = P.prefill(packed, prompt, use_cache=True)
    probs, tok, p1, margin = _step(out.logits[0, -1], first=True)
    neg_ent = float((probs * torch.log(probs.clamp_min(1e-12))).sum())
    toks, margins, logps, past = [tok], [margin], [float(np.log(max(p1, 1e-12)))], out.past_key_values
    while len(toks) < MAX_NEW and toks[-1] != EOS_ID:
        o = P.lm(inputs_embeds=P.emb(torch.tensor([[toks[-1]]], device=P.dev)), past_key_values=past, use_cache=True)
        past = o.past_key_values
        _, tok, p, mg = _step(o.logits[0, -1])
        toks.append(tok); margins.append(mg); logps.append(float(np.log(max(p, 1e-12))))
    sig = dict(first_margin=margin, first_p1=p1, first_neg_entropy=neg_ent,
               seq_min_margin=float(min(margins)), seq_mean_logp=float(np.mean(logps)))
    return P.decode_text(toks), toks, sig


@torch.inference_mode()
def run_signals():
    from exp.exp_paired_textvqa import load as load_question
    from models.llava_next import NextPaths
    OUT.mkdir(parents=True, exist_ok=True)
    plan = sorted(read_jsonl(E5_DIR / "plan.jsonl"), key=lambda q: q["order"])
    traces = {r["qid"]: r for r in read_jsonl(E5_DIR / "traces.jsonl")}
    images = {x["image_id"]: x for x in read_jsonl(MANIFEST_DIR / "textvqa_images.jsonl")}
    out_p = OUT / "signals.jsonl"
    done = {json.loads(l)["qid"] for l in open(out_p)} if out_p.exists() else set()
    todo = [q for q in plan if q["qid"] in traces and q["qid"] not in done]
    print(f"plan {len(plan)}, with E5 traces {len(traces)}, done {len(done)}, to run {len(todo)}", flush=True)
    P = NextPaths("cuda:0")
    with open(out_p, "a") as f:
        for k, q in enumerate(todo):
            img, pr, _ = load_question(P, q, images)
            packed, _ = P.base_encode(P.base_pixel(img))
            text, toks, sig = decode_with_signals(P, packed, pr)
            ref = traces[q["qid"]]
            rec = dict(qid=q["qid"], **sig, n_new=len(toks), text=text, same_text_as_e5=text == ref["base_text"],
                       margin_diff_vs_e5=abs(sig["first_margin"] - ref["base_margin"]))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(todo)}", flush=True)
    summarize()


def summarize():
    """AUROC of each signal for flagging a change (strict) and an accuracy drop (label-based); determinism check."""
    from sklearn.metrics import roc_auc_score
    sig = {r["qid"]: r for r in read_jsonl(OUT / "signals.jsonl")}
    rows = [r for r in read_jsonl(E5_DIR / "traces.jsonl") if r["qid"] in sig]
    y_change = np.array([r["changed"] for r in rows])
    y_drop = np.array([r["base_acc"] < r["full_acc"] for r in rows])
    same = np.mean([sig[r["qid"]]["same_text_as_e5"] for r in rows])
    md = max(sig[r["qid"]]["margin_diff_vs_e5"] for r in rows)
    L = ["# E8: confidence signals of the cheap path (TextVQA pilot DEV)", "",
         f"{len(rows)} questions. Determinism check against E5: cheap answer identical for {same:.1%}; largest first-margin "
         f"difference {md:.2e}.", "",
         "AUROC for flagging a loss (lower confidence = riskier). Strict: the cheap answer differs from the full answer; "
         "accuracy drop: the cheap answer is less accurate than the full answer.", "",
         "| Signal | read after | AUROC, strict | AUROC, accuracy drop |", "|---|---|---|---|"]
    for s in SIGNALS:
        v = np.array([sig[r["qid"]][s] for r in rows])
        where = "the whole cheap answer" if s.startswith("seq_") else "the cheap prefill"
        L.append(f"| {s} | {where} | {roc_auc_score(y_change, -v):.3f} | {roc_auc_score(y_drop, -v):.3f} |")
    (OUT / "signals.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


class ExpSignals(ExpBasic):
    def run(self):
        run_signals()
