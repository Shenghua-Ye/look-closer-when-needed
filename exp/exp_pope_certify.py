"""POPE certification and evaluation, with the main study's F2 functions imported unchanged (certify_one, eval_one,
policy_on, boot_saving). CPU only.

    python -m exp.exp_pope_certify --rehearse    # DEV rehearsal band only: the whole pipeline on pseudo-CAL / pseudo-TEST
    python -m exp.exp_pope_certify --certify     # step 1: CAL samples only; writes certificates_pope.json
    python -m exp.exp_pope_certify --evaluate    # step 2: needs the certificates; TEST metrics

Rows come from exp_pope_paired.py and use SugarCrepe's field names, so the cost model is the main study's for an A/B
task: accept = S0 + P + M; fallback = S0 + P + M + H (restart) or + H' (resume, loss 1[r != f]); direct full = H.
Rules: exp_f2_certify.RULES with restart and resume (main: group_ltt + resume at alpha = 5%); variant: the
accuracy-drop loss with restart. TEST adds descriptive per-setting slices (random / popular / adversarial): accuracy,
precision, recall, F1 for "yes" and yes-ratio of the full, cheap and certified answers.
"""
import argparse
import json

import numpy as np

from exp import exp_f2_certify as F2
from exp import pope_common as C
from exp.exp_frontier import ALPHAS

CONFIGS = [("signal-first_margin_loss-strict", "strict", ("restart", "resume")),
           ("signal-first_margin_loss-acc_drop", "acc_drop", ("restart",))]
MAIN = CONFIGS[0][0]


def arrays(rows, loss):
    """F2.arrays' SugarCrepe branch for POPE rows."""
    f = lambda k: np.array([r[k] for r in rows], float)
    acc = lambda key: np.array([r[key] == r["correct"] for r in rows], float)
    changed = np.array([bool(r["changed"]) for r in rows])
    acc_c, acc_f = acc("base_answer"), acc("full_answer")
    P, M = f("ms_P"), f("ms_M")
    return dict(n=len(rows), qid=np.array([r["qid"] for r in rows]), img=np.array([r["image_id"] for r in rows]),
                g=np.array([r["group"] for r in rows]), m=f("base_margin"), y=changed if loss == "strict" else acc_c < acc_f,
                yr=np.array([bool(r["resume_ne_full"]) for r in rows]), S0=f("ms_S_pre") + f("ms_S_enc"), P=P, M=M, C=P + M,
                H=f("ms_H_pre") + f("ms_H_fwd") + f("ms_H_score"), Hr=f("ms_R_pre") + f("ms_R_vis") + f("ms_R_cont") + f("ms_R_dec"),
                acc_c=acc_c, acc_f=acc_f, acc_r=acc("resume_answer"))


def certify_round(rows, paths_by_cfg, pg, pi, eligible_lock=None):
    runs = []
    for name, loss, fallbacks in CONFIGS:
        D = arrays(rows, loss)
        pos = {q: i for i, q in enumerate(D["qid"])}
        pgi = {g: np.array([pos[x["qid"]] for x in pg if x["group"] == g], int) for g in C.GROUPS}
        pii = np.array([pos[x["qid"]] for x in pi], int)
        allg = {g: np.flatnonzero(D["g"] == g) for g in C.GROUPS}
        paths = {g: [F2.lam_of(s["lam"]) for s in v["path"]] for g, v in paths_by_cfg[name].items()}
        for alpha in ALPHAS:
            for fb in fallbacks:
                for rule in F2.RULES:
                    c = F2.certify_one(D, C.GROUPS, paths, pgi, pii, allg, alpha, rule, fb)
                    if rule == "group_ltt" and eligible_lock is not None:
                        assert c["eligible"] == eligible_lock[f"{alpha:g}"]["eligible"], (alpha, c["eligible"])
                    c.update(config=name)
                    runs.append(c)
    return runs


def thresholds(c):
    return {g: (None if v is None else (v if isinstance(v, str) else float(v))) for g, v in c["thresholds"].items()}


def returned_answers(rows, D, th, fallback):
    _, _, act = F2.policy_on(D, np.arange(D["n"]), th, fallback)
    out = []
    for r, a in zip(rows, act):
        out.append(r["base_answer"] if a in (0, 2) else (r["resume_answer"] if (a == 3 and fallback == "resume") else r["full_answer"]))
    return out


def slice_metrics(answers, gold):
    a, g = np.array(answers) == "yes", np.array(gold) == "yes"
    tp, fp, fn = int((a & g).sum()), int((a & ~g).sum()), int((~a & g).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(n=len(a), accuracy=float((a == g).mean()), precision=prec, recall=rec,
                f1=(2 * prec * rec / (prec + rec) if prec + rec else 0.0), yes_ratio=float(a.mean()))


def evaluate_round(rows, runs, tg, ti, boot_seed=None, settings_of=None):
    """eval_one for always-cheap and every certified run; bootstrap and per-setting slices for the main rule."""
    out = []
    for name, loss, fallbacks in CONFIGS:
        D = arrays(rows, loss)
        pos = {q: i for i, q in enumerate(D["qid"])}
        pgi = {g: np.array([pos[x["qid"]] for x in tg if x["group"] == g], int) for g in C.GROUPS}
        pii = np.array([pos[x["qid"]] for x in ti], int)
        allq = np.arange(D["n"])
        out.append(F2.eval_one(D, C.GROUPS, pgi, pii, allq, {g: "cheap" for g in C.GROUPS}, "restart", None,
                               dict(config=name, rule="always_cheap", fallback="-", alpha=None)))
        for c in (x for x in runs if x["config"] == name):
            th = thresholds(c)
            rec = F2.eval_one(D, C.GROUPS, pgi, pii, allq, th, c["fallback"], c["alpha"],
                              dict(config=name, rule=c["rule"], fallback=c["fallback"], alpha=c["alpha"], eligible=c["eligible"]))
            if boot_seed is not None and name == MAIN and c["rule"] == "group_ltt":
                other = None
                if c["fallback"] == "resume":
                    rs = next(x for x in runs if x["config"] == name and x["rule"] == "group_ltt" and x["fallback"] == "restart" and x["alpha"] == c["alpha"])
                    other = (thresholds(rs), "restart")
                ci, dci = F2.boot_saving(D, allq, th, c["fallback"], boot_seed, other)
                rec.update(saving_ci=ci, saving_minus_restart_ci=dci)
            if settings_of is not None and name == MAIN and c["rule"] == "group_ltt" and c["fallback"] == "resume":
                ret = returned_answers(rows, D, th, "resume")
                rec["settings"] = {}
                for s in ("random", "popular", "adversarial"):
                    idx = [i for i, r in enumerate(rows) if s in settings_of[r["qid"]]]
                    gold = [rows[i]["correct"] for i in idx]
                    rec["settings"][s] = dict(full=slice_metrics([rows[i]["full_answer"] for i in idx], gold),
                                              cheap=slice_metrics([rows[i]["base_answer"] for i in idx], gold),
                                              certified=slice_metrics([ret[i] for i in idx], gold))
            out.append(rec)
    return out


def report_lines(title, res_rounds, note):
    L = [f"# {title}", "", note, ""]
    r0 = res_rounds[0]
    for alpha in ALPHAS:
        L += [f"### alpha = {alpha:.0%}, {MAIN}, round 0", "",
              "| rule | fallback | saving | worst group | groups > alpha | R_I | coverage | accuracy (full) | " + " | ".join(C.GROUPS) + " |",
              "|---|---|---|---|---|---|---|---|" + "---|" * len(C.GROUPS)]
        for x in r0["runs"]:
            if x["config"] != MAIN or x["alpha"] not in (None, alpha):
                continue
            g = " | ".join(f"{v['k']}/{v['n']}" + ("" if v["threshold"] is not None else " (full)") for v in x["groups"].values())
            s = f"{x['saving']:.1%}" + (f" [{x['saving_ci'][0]:.1%}, {x['saving_ci'][1]:.1%}]" if x.get("saving_ci") else "")
            L.append(f"| {x['rule']} | {x['fallback']} | {s} | {x['worst_group']} {x['worst']:.1%} | {', '.join(x['above_alpha']) or '-'} | "
                     f"{x['overall_RI']['rate']:.1%} | {x['coverage']:.0%} | {x['accuracy']:.1%} ({x['accuracy_full']:.1%}) | {g} |")
        L.append("")
    main_runs = [x for x in r0["runs"] if x["config"] == MAIN and x["rule"] == "group_ltt" and x["fallback"] == "resume" and x["alpha"] == C.ALPHA_MAIN]
    if main_runs and "settings" in main_runs[0]:
        L += ["### Per-setting slices (descriptive), round 0, main rule", "", "| setting | answers | n | accuracy | precision | recall | F1 | yes-ratio |", "|---|---|---|---|---|---|---|---|"]
        for s, d in main_runs[0]["settings"].items():
            for k, v in d.items():
                L.append(f"| {s} | {k} | {v['n']} | {v['accuracy']:.1%} | {v['precision']:.1%} | {v['recall']:.1%} | {v['f1']:.1%} | {v['yes_ratio']:.1%} |")
        L.append("")
    if len(res_rounds) > 1:
        L += ["### Rounds (main configuration): mean ± std", "", "| alpha | rule | fallback | saving | worst | rounds > alpha | coverage |", "|---|---|---|---|---|---|---|"]
        for alpha in ALPHAS:
            for rule in F2.RULES:
                for fb in ("restart", "resume"):
                    xs = [x for rr in res_rounds for x in rr["runs"] if x["config"] == MAIN and x["rule"] == rule and x["fallback"] == fb and x["alpha"] == alpha]
                    if len(xs) != len(res_rounds):
                        continue
                    sv, wr = np.array([x["saving"] for x in xs]), np.array([x["worst"] for x in xs])
                    L.append(f"| {alpha:.0%} | {rule} | {fb} | {sv.mean():.1%} ± {sv.std(ddof=1):.1%} | {wr.mean():.1%} ± {wr.std(ddof=1):.1%} | "
                             f"{sum(bool(x['above_alpha']) for x in xs)} | {np.mean([x['coverage'] for x in xs]):.0%} |")
        L.append("")
    return L


def settings_map():
    return {q["qid"]: q["settings"] for q in C.questions()}


def rehearse():
    """The whole certify -> evaluate path on the DEV rehearsal band (pseudo-CAL / pseudo-TEST halves). Code exercise only."""
    from exp import exp_pope_protocol as PR
    rows = PR.dev_rows("rehearsal")
    imgs = sorted({r["image_id"] for r in rows})
    u = np.random.default_rng([C.SEEDS["rehearsal"], C.CODE]).random(len(imgs))
    cal = [i for i, v in zip(imgs, u) if v < 0.5]
    test = [i for i, v in zip(imgs, u) if v >= 0.5]
    paths = {name: PR.candidate_paths(loss) for name, loss, _ in CONFIGS}
    pg, pi = PR.sample(cal, 0, C.SEEDS["rehearsal"])
    tg, ti = PR.sample(test, 0, C.SEEDS["rehearsal"] + 1)
    cal_rows = [r for r in rows if r["image_id"] in set(cal)]
    test_rows = [r for r in rows if r["image_id"] in set(test)]
    runs = certify_round(cal_rows, paths, pg, pi)
    res = [dict(round=0, runs=evaluate_round(test_rows, runs, tg, ti, boot_seed=C.SEEDS["bootstrap"], settings_of=settings_map()))]
    L = report_lines("POPE rehearsal on the DEV rehearsal band (code exercise only)", res,
                     f"Pseudo-CAL {len(cal)} images ({len(cal_rows)} questions), pseudo-TEST {len(test)} images ({len(test_rows)} questions). "
                     "These are DEV data; the numbers are not results and choose nothing. Small samples: most types are ineligible.")
    (C.DEV_DIR / "rehearsal_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def run_certify():
    lock = C.load_lock()
    C.P2_DIR.mkdir(parents=True, exist_ok=True)
    out = C.P2_DIR / "certificates_pope.json"
    if out.exists():
        raise SystemExit("certificates_pope.json exists: the certificates are written once")
    splits = json.loads((C.LOCK_DIR / "splits_pope.json").read_text())
    cal_s = json.loads((C.LOCK_DIR / "cal_sample_pope.json").read_text())
    tr = C.P1_DIR / "traces.jsonl"
    qmeta = C.questions()
    certs = dict(alphas=list(ALPHAS), rounds=[])
    for r, sp in enumerate(splits["rounds"]):
        cal = set(sp["CAL"])
        want = {q["qid"] for q in qmeta if q["image_id"] in cal}
        rows = [x for x in C.read_jsonl(tr) if x["qid"] in want]
        assert len(rows) == len(want) and all(x["image_id"] in cal for x in rows)
        runs = certify_round(rows, lock["paths"], cal_s[r]["per_group"], cal_s[r]["per_image"], lock["rounds"][r]["eligibility"])
        certs["rounds"].append(dict(round=r, n_cal_images=len(cal), n_rows=len(rows), runs=runs))
    out.write_text(json.dumps(certs, default=lambda o: o.item() if isinstance(o, np.generic) else o))
    md = ["# POPE certificates (CAL only)", "", "Written before any TEST metric.", "",
          "| round | config | fallback | alpha | rule | thresholds |", "|---|---|---|---|---|---|"]
    for rr in certs["rounds"]:
        for c in rr["runs"]:
            th = ", ".join(f"{g} {'full' if v is None else (v if isinstance(v, str) else f'{v:.4g}')}" for g, v in c["thresholds"].items())
            md.append(f"| {rr['round']} | {c['config']} | {c['fallback']} | {c['alpha']:.0%} | {c['rule']} | {th} |")
    (C.P2_DIR / "cal_report_pope.md").write_text("\n".join(md) + "\n")
    print("POPE certificates written")


def run_evaluate():
    from sklearn.metrics import roc_auc_score
    cp = C.P2_DIR / "certificates_pope.json"
    if not cp.exists():
        raise SystemExit("run --certify first")
    certs = json.loads(cp.read_text())
    tr = C.P1_DIR / "traces.jsonl"
    splits = json.loads((C.LOCK_DIR / "splits_pope.json").read_text())
    test_s = json.loads((C.LOCK_DIR / "test_sample_pope.json").read_text())
    rows_all = C.read_jsonl(tr)
    smap = settings_map()
    res = dict(rounds=[])
    for r, sp in enumerate(splits["rounds"]):
        test = set(sp["TEST"])
        rows = [x for x in rows_all if x["image_id"] in test]
        runs = next(x for x in certs["rounds"] if x["round"] == r)["runs"]
        res["rounds"].append(dict(round=r, n_test_questions=len(rows), runs=evaluate_round(
            rows, runs, test_s[r]["per_group"], test_s[r]["per_image"], boot_seed=(C.SEEDS["bootstrap"] if r == 0 else None),
            settings_of=(smap if r == 0 else None))))
        if r == 0:
            y = np.array([x["changed"] for x in rows])
            res["auroc_round0"] = dict(first_margin=float(roc_auc_score(y, -np.array([x["base_margin"] for x in rows]))),
                                       n=len(rows), changed_rate=float(y.mean()))
    (C.P2_DIR / "test_results_pope.json").write_text(json.dumps(res, indent=1, default=float))
    L = report_lines("POPE TEST results", res["rounds"],
                     f"AUROC of the forced-choice margin for a changed answer (round 0): {res['auroc_round0']['first_margin']:.3f} "
                     f"(n = {res['auroc_round0']['n']}, changed {res['auroc_round0']['changed_rate']:.1%}).")
    (C.P2_DIR / "test_report_pope.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearse", action="store_true")
    ap.add_argument("--certify", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    if a.rehearse:
        rehearse()
    if a.certify:
        run_certify()
    if a.evaluate:
        run_evaluate()


if __name__ == "__main__":
    main()
