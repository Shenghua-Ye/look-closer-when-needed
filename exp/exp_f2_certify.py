"""F2: certification on CAL (all five rounds, alphas, rules and variants), then TEST evaluation. CPU only.

    python -m exp.exp_f2_certify --certify      # step 1: reads only the CAL samples; writes certificates.json
    python -m exp.exp_f2_certify --evaluate     # step 2: needs certificates.json; TEST metrics

Inputs: results/F1_paired_caltest/<dataset>/traces.jsonl (one F1 run over all non-DEV questions), the F0 lock
(candidate paths, splits, CAL samples) and its TEST supplement (TEST samples). Nothing here changes the lock.

Policies (always probe; one threshold per group; a group without a threshold goes straight to the native full pass):
  restart  fallback = the native full pass from scratch; fallback loss 0
  resume   fallback = the resume continuation from the kept cheap cache; fallback loss 1[r_i != f_i]. The main
           certificate is this one.
Loss of an accepted request: 1[c_i != f_i] (strict; TextVQA after vqa-norm-v1), or, for the acc_drop variant,
1[acc(c_i) < acc(f_i)]. Cost model (exp_frontier): accept = S0 + C; fallback = S0 + P + M + H (restart) or S0 + P + M + H' (resume),
where seq_* signals count the whole cheap decode in P; direct full = H. All stage times are F1's measurements.

Rules (each certified and reported separately):
  group_ltt    ours: per-group fixed sequence along the group's candidate path, exact binomial p <= delta / |G_eligible|,
               eligibility n_g >= nmin(alpha, delta / |G|) (Bonferroni; joint 1 - delta)
  gpac         G-PAC-style: each group at level delta, eligibility n_g >= nmin(alpha, delta) (each group 1 - delta)
  shared_iut   one threshold along the shared candidate path '*', every eligible group (main eligibility) at delta (IUT)
  overall_ltt  CALM-style: one threshold along '*', the per-image CAL sample at delta (overall risk only)
  group_crc    HG-CRC-style: per group, the loosest path setting with (k + 1) / (n + 1) <= alpha (expected risk)
  empirical    per group, the cheapest path setting whose CAL rate over all CAL questions of the group is <= alpha
  group_fixed  per-type fixed path: the cheap answer or the full pass per group, certified like group_ltt
Configurations: main (SugarCrepe A/B margin, TextVQA seq_min_margin; strict loss) with both fallbacks;
variants with the restart fallback: TextVQA first_margin (strict), both datasets acc_drop.

TEST: per group n, k, k/n and the exact two-sided 95% Clopper-Pearson interval on the fixed
per_group TEST sample; the per-image overall rate R_I (per_image sample); coverage; accuracy; replayed latency saving
over all TEST questions (question-weighted, deployment view) with a paired 95% image-cluster bootstrap for round 0;
mean / p50 / p95 replayed latency. Rounds 1-4 describe split sensitivity. Online timing is exp_f3_timing.py.
Outputs: results/F2_certify_evaluate/: certificates.json, cal_report.md, test_results.json,
test_report.md, test_groups.csv.
"""
import argparse
import json

import numpy as np
from scipy.stats import beta, binom

from exp.exp_frontier import ALPHAS, DELTA, nmin
from root import RESULTS_DIR, ROOT_DIR
from utils.tools import read_jsonl

F1 = RESULTS_DIR / "F1_paired_caltest"
LOCK_DIR = RESULTS_DIR / "F0_protocol_lock"
OUT = RESULTS_DIR / "F2_certify_evaluate"
DATASETS = ("sugarcrepe", "textvqa")
MAIN = {"sugarcrepe": ("first_margin", "strict"), "textvqa": ("seq_min_margin", "strict")}
MAIN_ALPHA = {"sugarcrepe": 0.05, "textvqa": 0.10}
VARIANTS = {"sugarcrepe": [("first_margin", "acc_drop")], "textvqa": [("first_margin", "strict"), ("seq_min_margin", "acc_drop")]}
RULES = ("group_ltt", "gpac", "shared_iut", "overall_ltt", "group_crc", "empirical", "group_fixed")
BOOT, BOOT_SEED = 2000, 20261106


def lock():
    return json.loads((LOCK_DIR / "protocol_lock.json").read_text())


def configs(ds):
    """(name, signal, loss, fallbacks)."""
    s, l = MAIN[ds]
    out = [(f"signal-{s}_loss-{l}", s, l, ("restart", "resume"))]
    out += [(f"signal-{s2}_loss-{l2}", s2, l2, ("restart",)) for s2, l2 in VARIANTS[ds]]
    return out


def load_rows(ds, qids=None):
    """F1 rows, only those in qids (the CAL step never keeps a row outside the CAL samples)."""
    keep = None if qids is None else set(qids)
    rows = []
    with open(F1 / ds / "traces.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if keep is None or r["qid"] in keep:
                rows.append(r)
    return rows


def arrays(ds, rows, signal, loss):
    """Per-question arrays in the cost model of exp_frontier, for one configuration."""
    f = lambda k: np.array([r[k] for r in rows], float)
    S0 = f("ms_S_pre") + f("ms_S_enc")
    Hr = f("ms_R_pre") + f("ms_R_vis") + f("ms_R_cont") + f("ms_R_dec")
    if ds == "sugarcrepe":
        H, P, M, L = f("ms_H_pre") + f("ms_H_fwd") + f("ms_H_score"), f("ms_P"), f("ms_M"), np.zeros(len(rows))
        m = f("base_margin")
        acc_c = np.array([r["base_answer"] == r["correct"] for r in rows], float)
        acc_f = np.array([r["full_answer"] == r["correct"] for r in rows], float)
        acc_r = np.array([r["resume_answer"] == r["correct"] for r in rows], float)
    else:
        H, P, M, L = f("ms_H"), f("ms_P"), np.zeros(len(rows)), f("ms_D")
        m = f(signal)
        if signal.startswith("seq_"):  # read after the whole cheap answer: a fallback also wastes the decode
            P, L = P + L, np.zeros(len(rows))
        acc_c, acc_f = f("base_acc"), f("full_acc")
        acc_r = acc_f  # resume answers are only used with the strict loss; accuracy of r is computed below
    changed = np.array([bool(r["changed"]) for r in rows])
    y = changed if loss == "strict" else acc_c < acc_f
    yr = np.array([bool(r["resume_ne_full"]) for r in rows])
    return dict(n=len(rows), qid=np.array([r["qid"] for r in rows]), img=np.array([r["image_id"] for r in rows]),
                g=np.array([r["group"] for r in rows]), m=m, y=y, yr=yr, S0=S0, P=P, M=M, C=P + M + L, H=H, Hr=Hr,
                acc_c=acc_c, acc_f=acc_f, acc_r=acc_r)


def outcome(D, idx, th, fallback):
    """cost, loss, action (0 cheap direct, 2 accept, 3 fallback, 4 safe full) of requests idx under setting th."""
    idx = np.asarray(idx, int)
    if th is None:
        return D["H"][idx], np.zeros(len(idx), bool), np.full(len(idx), 4)
    if th == "cheap":  # per-type fixed path: the cheap answer for the whole group
        return D["S0"][idx] + D["C"][idx], D["y"][idx].copy(), np.full(len(idx), 0)
    acc = D["m"][idx] > th
    fb_cost = D["H"][idx] if fallback == "restart" else D["Hr"][idx]
    fb_loss = np.zeros(len(idx), bool) if fallback == "restart" else D["yr"][idx]
    cost = D["S0"][idx] + np.where(acc, D["C"][idx], D["P"][idx] + D["M"][idx] + fb_cost)
    return cost, np.where(acc, D["y"][idx], fb_loss), np.where(acc, 2, 3)


def lam_of(x):
    return {"-inf": -np.inf, "inf": np.inf}.get(x, x) if isinstance(x, str) else float(x)


# ------------------------------------------------------------------------------------------------ CAL
def fixed_sequence(D, path, tests, alpha, level, fallback, trace):
    """Walk the path from the safest setting; a setting passes if p <= level in every test set; stop at the first
    failure; keep the last passing setting (None = full pass). trace collects (lam, [(n, k, p)]) per tested setting."""
    chosen = None
    for lam in path:
        res = []
        for idx in tests:
            k = int(outcome(D, idx, lam, fallback)[1].sum())
            res.append((len(idx), k, float(binom.cdf(k, len(idx), alpha))))
        trace.append(dict(lam=lam, tests=res))
        if any(p > level for _, _, p in res):
            break
        chosen = lam
    return chosen


def certify_one(D, groups, paths, per_group_idx, per_image_idx, all_by_group, alpha, rule, fallback):
    """Thresholds {group: lam | 'cheap' | None} and the CAL record of one rule."""
    G = len(groups)
    rec = dict(rule=rule, fallback=fallback, alpha=alpha)
    out, trace = {g: None for g in groups}, {}
    n = {g: len(per_group_idx[g]) for g in groups}
    if rule in ("group_ltt", "group_fixed"):
        elig = [g for g in groups if n[g] >= nmin(alpha, DELTA / G)]
        level = DELTA / max(1, len(elig))
        for g in elig:
            trace[g] = []
            path = ["cheap"] if rule == "group_fixed" else paths[g]
            out[g] = fixed_sequence(D, path, [per_group_idx[g]], alpha, level, fallback, trace[g])
        rec.update(eligible=elig, level=level)
    elif rule == "gpac":
        elig = [g for g in groups if n[g] >= nmin(alpha, DELTA)]
        for g in elig:
            trace[g] = []
            out[g] = fixed_sequence(D, paths[g], [per_group_idx[g]], alpha, DELTA, fallback, trace[g])
        rec.update(eligible=elig, level=DELTA)
    elif rule == "shared_iut":
        elig = [g for g in groups if n[g] >= nmin(alpha, DELTA / G)]
        trace["*"] = []
        th = fixed_sequence(D, paths["*"], [per_group_idx[g] for g in elig], alpha, DELTA, fallback, trace["*"]) if elig else None
        out = {g: (th if g in elig else None) for g in groups}
        rec.update(eligible=elig, level=DELTA)
    elif rule == "overall_ltt":
        ok = len(per_image_idx) >= nmin(alpha, DELTA)
        trace["*"] = []
        th = fixed_sequence(D, paths["*"], [per_image_idx], alpha, DELTA, fallback, trace["*"]) if ok else None
        out = {g: th for g in groups}
        rec.update(eligible=list(groups) if ok else [], level=DELTA, n_per_image=len(per_image_idx))
    elif rule == "group_crc":
        for g in groups:
            trace[g] = []
            for lam in paths[g]:
                k = int(outcome(D, per_group_idx[g], lam, fallback)[1].sum())
                trace[g].append(dict(lam=lam, n=n[g], k=k, crc=(k + 1) / (n[g] + 1)))
                if (k + 1) / (n[g] + 1) > alpha:
                    break
                out[g] = lam
        rec.update(eligible=list(groups))
    elif rule == "empirical":
        for g in groups:
            trace[g] = []
            for lam in paths[g]:
                idx = all_by_group[g]
                r = float(outcome(D, idx, lam, fallback)[1].mean()) if len(idx) else 1.0
                trace[g].append(dict(lam=lam, n=len(idx), rate=r))
                if len(idx) and r <= alpha:
                    out[g] = lam
        rec.update(eligible=list(groups))
    rec.update(n_cal=n, thresholds={g: (None if v is None else (v if isinstance(v, str) else float(v))) for g, v in out.items()},
               trace=trace)
    return rec


def run_certify():
    L = lock()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "certificates.json").exists():
        raise SystemExit("certificates.json exists: the certificates are written once (delete deliberately only before any "
                         "TEST metric has been computed)")
    certs = dict(delta=DELTA, alphas=list(ALPHAS), datasets={})
    md = ["# F2 step 1: certificates on CAL (all rounds, alphas, rules)", "",
          "Only CAL samples were read; written before any TEST metric. Thresholds per group; `full` = no certificate "
          "(direct full pass).", ""]
    for ds in DATASETS:
        groups = L["groups"][ds]
        splits = json.loads((LOCK_DIR / f"splits_{ds}.json").read_text())
        cal_s = json.loads((LOCK_DIR / f"cal_sample_{ds}.json").read_text())
        qmeta = read_jsonl(ROOT_DIR / "data" / "manifests" / f"{ds}_questions.jsonl")
        certs["datasets"][ds] = []
        for r, sp in enumerate(splits["rounds"]):
            cal_imgs = set(sp["CAL"])
            pg, pi = cal_s[r]["per_group"], cal_s[r]["per_image"]
            all_cal = [q["qid"] for q in qmeta if q["image_id"] in cal_imgs]
            rows = load_rows(ds, set(all_cal))
            assert all(r_["image_id"] in cal_imgs for r_ in rows) and len(rows) == len(all_cal)
            rec_round = dict(round=r, n_cal_images=len(cal_imgs), n_rows=len(rows), runs=[])
            for name, signal, loss, fallbacks in configs(ds):
                D = arrays(ds, rows, signal, loss)
                pos = {q: i for i, q in enumerate(D["qid"])}
                per_group_idx = {g: np.array([pos[x["qid"]] for x in pg if x["group"] == g], int) for g in groups}
                per_image_idx = np.array([pos[x["qid"]] for x in pi], int)
                all_by_group = {g: np.flatnonzero(D["g"] == g) for g in groups}
                paths = {g: [lam_of(s["lam"]) for s in v["path"]] for g, v in L["datasets"][ds]["paths"][name].items()}
                for alpha in ALPHAS:
                    el_lock = L["datasets"][ds]["rounds"][r]["eligibility"][f"{alpha:g}"]["eligible"]
                    for fb in fallbacks:
                        for rule in RULES:
                            c = certify_one(D, groups, paths, per_group_idx, per_image_idx, all_by_group, alpha, rule, fb)
                            if rule == "group_ltt":
                                assert c["eligible"] == el_lock, (ds, r, alpha, c["eligible"], el_lock)
                            c.update(config=name)
                            rec_round["runs"].append(c)
            certs["datasets"][ds].append(rec_round)
        # report: the main rule, round 0 and all rounds' thresholds
        md += [f"## {ds}", "", "| round | config | fallback | alpha | rule | thresholds |", "|---|---|---|---|---|---|"]
        for rr in certs["datasets"][ds]:
            for c in rr["runs"]:
                if c["rule"] in ("group_ltt",) or rr["round"] == 0:
                    th = ", ".join(f"{g} {'full' if v is None else (v if isinstance(v, str) else f'{v:.4g}')}" for g, v in c["thresholds"].items())
                    md.append(f"| {rr['round']} | {c['config']} | {c['fallback']} | {c['alpha']:.0%} | {c['rule']} | {th} |")
        md.append("")
    js = json.dumps(certs, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    (OUT / "certificates.json").write_text(js)
    (OUT / "cal_report.md").write_text("\n".join(md) + "\n")
    print(f"certificates written ({sum(len(rr['runs']) for d in certs['datasets'].values() for rr in d)} rule runs)")


# ------------------------------------------------------------------------------------------------ TEST
def cp(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def policy_on(D, idx, thresholds, fallback):
    """cost, loss, action of requests idx (any groups) under per-group thresholds."""
    idx = np.asarray(idx, int)
    cost, loss, act = np.zeros(len(idx)), np.zeros(len(idx), bool), np.zeros(len(idx), int)
    for g, th in thresholds.items():
        sel = D["g"][idx] == g
        if sel.any():
            c, l, a = outcome(D, idx[sel], th, fallback)
            cost[sel], loss[sel], act[sel] = c, l, a
    return cost, loss, act


def returned_acc(D, idx, act, fallback):
    acc = np.where(np.isin(act, (0, 2)), D["acc_c"][idx], np.where(act == 3, D["acc_r"][idx] if fallback == "resume" else D["acc_f"][idx], D["acc_f"][idx]))
    return float(acc.mean())


def boot_saving(D, idx, thresholds, fallback, seed, other=None):
    """Paired 95% image-cluster bootstrap of the saving (and of saving minus another policy's, if given)."""
    idx = np.asarray(idx, int)
    cost = policy_on(D, idx, thresholds, fallback)[0]
    H = D["H"][idx]
    c2 = policy_on(D, idx, other[0], other[1])[0] if other else None
    imgs, inv = np.unique(D["img"][idx], return_inverse=True)
    sc, sh = np.bincount(inv, cost), np.bincount(inv, H)
    s2 = np.bincount(inv, c2) if other else None
    rng = np.random.default_rng(seed)
    sav, diff = [], []
    for _ in range(BOOT):
        w = np.bincount(rng.integers(len(imgs), size=len(imgs)), minlength=len(imgs))
        a = 1 - (w @ sc) / (w @ sh)
        sav.append(a)
        if other:
            diff.append(a - (1 - (w @ s2) / (w @ sh)))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(sav), (q(diff) if other else None)


def run_evaluate():
    from sklearn.metrics import roc_auc_score
    L = lock()
    cpath = OUT / "certificates.json"
    if not cpath.exists():
        raise SystemExit("run --certify first: TEST metrics only after every certificate is written")
    certs = json.loads(cpath.read_text())
    tests = {ds: json.loads((LOCK_DIR / f"test_sample_{ds}.json").read_text()) for ds in DATASETS}
    results = dict(datasets={})
    for ds in DATASETS:
        groups = L["groups"][ds]
        splits = json.loads((LOCK_DIR / f"splits_{ds}.json").read_text())
        rows_all = load_rows(ds)
        results["datasets"][ds] = dict(rounds=[], auroc={})
        for r, sp in enumerate(splits["rounds"]):
            test_imgs = set(sp["TEST"])
            rows = [x for x in rows_all if x["image_id"] in test_imgs]
            pg, pi = tests[ds][r]["per_group"], tests[ds][r]["per_image"]
            rr = dict(round=r, n_test_questions=len(rows), n_test_images=len(test_imgs), runs=[])
            for name, signal, loss, fallbacks in configs(ds):
                D = arrays(ds, rows, signal, loss)
                if ds == "textvqa":  # accuracy of the resumed answer (strict-loss runs only)
                    from utils.vqa_eval import accuracy
                    qa = {q["qid"]: q["answers"] for q in read_jsonl(ROOT_DIR / "data" / "manifests" / "textvqa_questions.jsonl")
                          if q["image_id"] in test_imgs}
                    D["acc_r"] = np.array([accuracy(x["resume_text"], qa[x["qid"]]) for x in rows], float)
                pos = {q: i for i, q in enumerate(D["qid"])}
                pgi = {g: np.array([pos[x["qid"]] for x in pg if x["group"] == g], int) for g in groups}
                pii = np.array([pos[x["qid"]] for x in pi], int)
                allq = np.arange(D["n"])
                if r == 0 and name == configs(ds)[0][0]:
                    results["datasets"][ds]["auroc"] = auroc_block(ds, rows, D, roc_auc_score)
                # always-cheap and always-full references
                for ref in ("always_cheap",):
                    th = {g: "cheap" for g in groups}
                    rr["runs"].append(eval_one(D, groups, pgi, pii, allq, th, "restart", None, dict(config=name, rule=ref, fallback="-", alpha=None)))
                cr = [c for x in certs["datasets"][ds] if x["round"] == r for c in x["runs"] if c["config"] == name]
                for c in cr:
                    th = {g: (None if v is None else (v if isinstance(v, str) else float(v))) for g, v in c["thresholds"].items()}
                    rec = eval_one(D, groups, pgi, pii, allq, th, c["fallback"], c["alpha"],
                                   dict(config=name, rule=c["rule"], fallback=c["fallback"], alpha=c["alpha"], eligible=c["eligible"]))
                    if r == 0 and c["rule"] == "group_ltt" and name == configs(ds)[0][0]:
                        other = ({g: (None if v is None else float(v)) for g, v in next(
                            x for x in cr if x["rule"] == "group_ltt" and x["fallback"] == "restart" and x["alpha"] == c["alpha"])["thresholds"].items()}, "restart")
                        ci, dci = boot_saving(D, allq, th, c["fallback"], BOOT_SEED, other if c["fallback"] == "resume" else None)
                        rec.update(saving_ci=ci, saving_minus_restart_ci=dci)
                    rr["runs"].append(rec)
            results["datasets"][ds]["rounds"].append(rr)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_results.json").write_text(json.dumps(results, indent=1, default=float))
    write_test_report(results, L)


def eval_one(D, groups, pgi, pii, allq, th, fallback, alpha, head):
    rec = dict(head, groups={})
    for g in groups:
        idx = pgi[g]
        n = len(idx)
        k = int(policy_on(D, idx, {g: th[g]}, fallback)[1].sum()) if n else 0
        lo, hi = cp(k, n) if n else (None, None)
        rec["groups"][g] = dict(n=n, k=k, rate=(k / n if n else None), ci=(lo, hi), threshold=th[g])
    rates = {g: v["rate"] for g, v in rec["groups"].items() if v["rate"] is not None}
    rec["worst_group"] = max(rates, key=rates.get)
    rec["worst"] = rates[rec["worst_group"]]
    rec["above_alpha"] = [g for g, v in rates.items() if alpha is not None and v > alpha]
    k_i = int(policy_on(D, pii, th, fallback)[1].sum())
    rec["overall_RI"] = dict(n=len(pii), k=k_i, rate=k_i / len(pii), ci=cp(k_i, len(pii)))
    cost, loss, act = policy_on(D, allq, th, fallback)
    rec["RQ_all_questions"] = float(loss.mean())
    rec["coverage"] = float(np.mean([th[g] is not None for g in D["g"][allq]]))
    rec["saving"] = float(1 - cost.sum() / D["H"][allq].sum())
    rec["latency_ms"] = dict(mean=float(cost.mean()), p50=float(np.median(cost)), p95=float(np.percentile(cost, 95)))
    rec["full_latency_ms"] = dict(mean=float(D["H"][allq].mean()), p50=float(np.median(D["H"][allq])), p95=float(np.percentile(D["H"][allq], 95)))
    rec["actions"] = {k: float(np.mean(act == v)) for k, v in (("cheap", 0), ("accept", 2), ("fallback", 3), ("full", 4))}
    rec["accuracy"] = returned_acc(D, allq, act, fallback)
    rec["accuracy_full"] = float(D["acc_f"][allq].mean())
    fb = act == 3
    if fb.any():
        over = (cost - D["H"][allq])[fb]
        rec["fallback_extra_ms"] = dict(mean=float(over.mean()), p50=float(np.median(over)), p95=float(np.percentile(over, 95)), max=float(over.max()))
        rec["fallback_mismatch"] = int((D["yr"][allq] & fb).sum()) if fallback == "resume" else 0
    return rec


def auroc_block(ds, rows, D, roc_auc_score):
    y = np.array([bool(r["changed"]) for r in rows])
    sig = ["first_margin"] if ds == "sugarcrepe" else ["first_margin", "first_p1", "first_neg_entropy", "seq_min_margin", "seq_mean_logp"]
    key = {"first_margin": "base_margin"} if ds == "sugarcrepe" else {}
    out = {s: float(roc_auc_score(y, -np.array([r[key.get(s, s)] for r in rows], float))) for s in sig}
    if ds == "textvqa":
        out["last_step_margin"] = float(roc_auc_score(y, -np.array([r["step_margins"][-1] for r in rows])))
        out["mean_step_margin"] = float(roc_auc_score(y, -np.array([np.mean(r["step_margins"]) for r in rows])))
    out["n"], out["changed_rate"] = len(rows), float(y.mean())
    return out


def write_test_report(res, L):
    md = ["# F2 step 2: TEST results", "", "The certificates were written before any TEST metric. "
          "Risk = per-group image-weighted rate on the fixed per_group TEST sample (k/n, exact 95% Clopper-Pearson). "
          "Saving = replayed latency saving over all TEST questions of the round (F1 stage times, cost model of exp_frontier).", ""]
    rows_csv = ["dataset,round,config,rule,fallback,alpha,group,n,k,rate,ci_lo,ci_hi,threshold"]
    for ds, d in res["datasets"].items():
        groups = L["groups"][ds]
        md += [f"## {ds}", "", "AUROC of the cheap confidence for a changed answer (round-0 TEST, all questions): " +
               ", ".join(f"{k} {v:.3f}" for k, v in d["auroc"].items() if k not in ("n", "changed_rate")) +
               f" (n = {d['auroc']['n']}, changed {d['auroc']['changed_rate']:.1%}).", ""]
        main = configs(ds)[0][0]
        for alpha in ALPHAS:
            md += [f"### alpha = {alpha:.0%}, {main}, round 0", "",
                   "| rule | fallback | saving | worst group | groups > alpha | R_I [CI] | coverage | accuracy (full) | " + " | ".join(groups) + " |",
                   "|---|---|---|---|---|---|---|---|" + "---|" * len(groups)]
            r0 = d["rounds"][0]
            for x in r0["runs"]:
                if x["config"] != main or (x["alpha"] not in (None, alpha)):
                    continue
                g = " | ".join(f"{v['k']}/{v['n']}" + ("" if v["threshold"] is not None else " (full)") for v in x["groups"].values())
                s = f"{x['saving']:.1%}" + (f" [{x['saving_ci'][0]:.1%}, {x['saving_ci'][1]:.1%}]" if x.get("saving_ci") else "")
                md.append(f"| {x['rule']} | {x['fallback']} | {s} | {x['worst_group']} {x['worst']:.1%} | {', '.join(x['above_alpha']) or '-'} | "
                          f"{x['overall_RI']['rate']:.1%} [{x['overall_RI']['ci'][0]:.1%}, {x['overall_RI']['ci'][1]:.1%}] | {x['coverage']:.0%} | "
                          f"{x['accuracy']:.1%} ({x['accuracy_full']:.1%}) | {g} |")
            md.append("")
        md += ["### Five rounds (main configuration): mean ± std over rounds", "",
               "| alpha | rule | fallback | saving | worst group rate | rounds with a group > alpha | coverage |", "|---|---|---|---|---|---|---|"]
        for alpha in ALPHAS:
            for rule in RULES:
                for fb in ("restart", "resume"):
                    xs = [x for rr in d["rounds"] for x in rr["runs"] if x["config"] == main and x["rule"] == rule and x["fallback"] == fb and x["alpha"] == alpha]
                    if len(xs) != 5:
                        continue
                    sv, wr = np.array([x["saving"] for x in xs]), np.array([x["worst"] for x in xs])
                    md.append(f"| {alpha:.0%} | {rule} | {fb} | {sv.mean():.1%} ± {sv.std(ddof=1):.1%} | {wr.mean():.1%} ± {wr.std(ddof=1):.1%} | "
                              f"{sum(bool(x['above_alpha']) for x in xs)} | {np.mean([x['coverage'] for x in xs]):.0%} |")
        md.append("")
        for rr in d["rounds"]:
            for x in rr["runs"]:
                for g, v in x["groups"].items():
                    rows_csv.append(f"{ds},{rr['round']},{x['config']},{x['rule']},{x['fallback']},{x['alpha']},{g},{v['n']},{v['k']},"
                                    f"{'' if v['rate'] is None else v['rate']},{v['ci'][0]},{v['ci'][1]},{v['threshold']}")
    (OUT / "test_report.md").write_text("\n".join(md) + "\n")
    (OUT / "test_groups.csv").write_text("\n".join(rows_csv) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--certify", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    if a.certify:
        run_certify()
    if a.evaluate:
        run_evaluate()


if __name__ == "__main__":
    main()
