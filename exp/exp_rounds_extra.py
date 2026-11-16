"""1,000 calibration/test rounds. CPU only: it replays the paired runs of F1 / P1 (no model is run).

    python -m exp.exp_rounds_extra --rounds 1000          # about 15 minutes, one process per dataset
    python -m exp.exp_rounds_extra --summarize            # rebuild summary*.json and report.md from rounds.json.gz
    python -m exp.exp_rounds_extra --fixed                # always cheap and VisionZip over the same rounds (needs F7)

Round r follows the protocol's random streams: round 0 is the manifest split; round r >= 1 redraws the non-DEV images
with default_rng([20261101, dataset, r]) (CAL if u < 0.5, else TEST); the CAL and TEST samples (one question per
image-type pair, one per image) use the seeds 20261102 and 20261103. Rounds 0-4 are the rounds stored in F2 / P2 and
are checked against them below. DEV, the type rules, the signals and the candidate paths are the fixed ones.

Per round, dataset, alpha in {5, 10, 20}% and rule (ours = group_ltt with resume and restart; gpac, shared_iut,
overall_ltt, group_crc, empirical with resume): thresholds from CAL only, then on TEST
- `saving` (replayed from the recorded stage times, all TEST questions), `worst` (highest per-type rate on the TEST
  sample), `exceed_test` (some type's TEST-sample rate > alpha);
- `exceed_heldout`: some type's held-out risk > alpha, the risk of a type computed on ALL TEST questions with images
  weighted equally (per image the mean loss over its questions of that type, then the mean over images);
- `exceed_pool`: the same on all CAL and TEST images together (includes CAL, so optimistic).

Output: results/F6_rounds_extra/{rounds.json.gz, summary.json (main alpha, resume), summary_alpha.json, report.md}."""
import argparse
import collections
import gzip
import json
from multiprocessing import Pool

import numpy as np

from exp import exp_f2_certify as F2
from root import RESULTS_DIR

OUT = RESULTS_DIR / "F6_rounds_extra"
RULES = ("group_ltt", "gpac", "shared_iut", "overall_ltt", "group_crc", "empirical")
RUNS = [(r, "resume") for r in RULES] + [("group_ltt", "restart")]
ALPHA_SET = (0.05, 0.10, 0.20)
MAIN_ALPHA = {"sugarcrepe": 0.05, "textvqa": 0.10, "pope": 0.05}
NAME = {"group_ltt": "Ours", "gpac": "G-PAC-style", "shared_iut": "Shared (IUT)", "overall_ltt": "Overall LTT",
        "group_crc": "Per-type CRC", "empirical": "Empirical"}


def key(alpha, rule, fb):
    return f"{alpha:g}|{rule}|{fb}"


# ------------------------------------------------------------------------------------------------ datasets
def build_spec(ds):
    if ds == "pope":
        from exp import exp_pope_certify as PC
        from exp import exp_pope_protocol as PP
        from exp import pope_common as C
        lock = C.load_lock()
        imgs = sorted(C.images().values(), key=lambda x: x["image_id"])
        D = PC.arrays(C.read_jsonl(C.P1_DIR / "traces.jsonl"), "strict")
        paths = {g: [F2.lam_of(s["lam"]) for s in v["path"]] for g, v in lock["paths"][PC.MAIN].items()}
        return dict(groups=list(C.GROUPS), D=D, paths=paths, imgs=imgs, seed=C.SEEDS["rounds"], code=C.CODE, config=PC.MAIN,
                    cal=lambda c, r: PP.sample(c, r, C.SEEDS["cal_sample"]), test=lambda t, r: PP.sample(t, r, C.SEEDS["test_sample"]))
    from data_provider import datasets
    from exp import exp_protocol as PR
    from exp import exp_protocol_test as PT
    from utils.tools import read_jsonl
    L = F2.lock()
    name = f"signal-{F2.MAIN[ds][0]}_loss-{F2.MAIN[ds][1]}"
    imgs = sorted(read_jsonl(datasets[ds]["images"]), key=lambda x: x["image_id"])
    D = F2.arrays(ds, F2.load_rows(ds), *F2.MAIN[ds])
    paths = {g: [F2.lam_of(s["lam"]) for s in v["path"]] for g, v in L["datasets"][ds]["paths"][name].items()}
    return dict(groups=L["groups"][ds], D=D, paths=paths, imgs=imgs, seed=PR.ROUND_SEED, code=PR.CODE[ds], config=name,
                cal=lambda c, r: PR.cal_sample(ds, c, r), test=lambda t, r: PT.test_sample(ds, t, r))


def split(spec, r):
    imgs = spec["imgs"]
    if r == 0:
        return [x["image_id"] for x in imgs if x["split"] == "CAL"], [x["image_id"] for x in imgs if x["split"] == "TEST"]
    rest = [x["image_id"] for x in imgs if x["split"] != "DEV"]
    u = np.random.default_rng([spec["seed"], spec["code"], r]).random(len(rest))
    return [i for i, v in zip(rest, u) if v < 0.5], [i for i, v in zip(rest, u) if v >= 0.5]


# ------------------------------------------------------------------------------------------------ one round
def risk(D, idx, g, th, fb):
    """Image-weighted risk of type g on questions idx: mean over images of the mean loss over that image's questions."""
    sel = idx[D["g"][idx] == g]
    if not len(sel):
        return None
    loss = F2.outcome(D, sel, th, fb)[1].astype(float)
    per_img = collections.defaultdict(list)
    for im, l in zip(D["img"][sel], loss):
        per_img[im].append(l)
    return float(np.mean([np.mean(v) for v in per_img.values()]))


def one_round(spec, r):
    D, groups = spec["D"], spec["groups"]
    cal_imgs, test_imgs = split(spec, r)
    pos = {q: i for i, q in enumerate(D["qid"])}
    pg, pi = spec["cal"](cal_imgs, r)
    tg, ti = spec["test"](test_imgs, r)
    cal_set, test_set = set(cal_imgs), set(test_imgs)
    is_cal = np.array([im in cal_set for im in D["img"]])
    is_test = np.array([im in test_set for im in D["img"]])
    pgi = {g: np.array([pos[x["qid"]] for x in pg if x["group"] == g], int) for g in groups}
    pii = np.array([pos[x["qid"]] for x in pi], int)
    allg = {g: np.flatnonzero((D["g"] == g) & is_cal) for g in groups}
    tgi = {g: np.array([pos[x["qid"]] for x in tg if x["group"] == g], int) for g in groups}
    tii = np.array([pos[x["qid"]] for x in ti], int)
    test_q, pool_q = np.flatnonzero(is_test), np.flatnonzero(is_test | is_cal)
    cheap = F2.eval_one(D, groups, tgi, tii, test_q, {g: "cheap" for g in groups}, "restart", None, {})
    out = {"_cheap_saving": cheap["saving"]}
    for alpha in ALPHA_SET:
        for rule, fb in RUNS:
            c = F2.certify_one(D, groups, spec["paths"], pgi, pii, allg, alpha, rule, fb)
            th = {g: (None if v is None else (v if isinstance(v, str) else float(v))) for g, v in c["thresholds"].items()}
            e = F2.eval_one(D, groups, tgi, tii, test_q, th, fb, alpha, {})
            held = {g: (0.0 if th[g] is None else risk(D, test_q, g, th[g], fb)) for g in groups}
            pool = {g: (0.0 if th[g] is None else risk(D, pool_q, g, th[g], fb)) for g in groups}
            out[key(alpha, rule, fb)] = dict(
                thresholds=th, eligible=c["eligible"], saving=e["saving"], worst=e["worst"], worst_group=e["worst_group"],
                exceed_test=len(e["above_alpha"]) > 0, coverage=e["coverage"], heldout=held,
                exceed_heldout=any(v is not None and v > alpha for v in held.values()), pool=pool,
                exceed_pool=any(v is not None and v > alpha for v in pool.values()))
    return out


def run_dataset(args):
    ds, R = args
    spec = build_spec(ds)
    rounds = [one_round(spec, r) for r in range(R)]
    print(f"{ds}: {R} rounds", flush=True)
    return ds, dict(config=spec["config"], rounds=rounds)


# ------------------------------------------------------------------------------------------------ checks and summaries
def check_against_reports(results):
    """Rounds 0-4 must reproduce the rounds stored in F2 / P2 (thresholds, saving, worst) for every alpha and rule."""
    T = json.loads((F2.OUT / "test_results.json").read_text())
    TP = json.loads((RESULTS_DIR / "P2_pope_certify_evaluate" / "test_results_pope.json").read_text())
    n = 0
    for ds, res in results.items():
        for r in range(min(5, len(res["rounds"]))):
            ref_runs = TP["rounds"][r]["runs"] if ds == "pope" else T["datasets"][ds]["rounds"][r]["runs"]
            for alpha in ALPHA_SET:
                for rule, fb in RUNS:
                    ref = [x for x in ref_runs if x["rule"] == rule and x["fallback"] == fb and x["config"] == res["config"]
                           and x["alpha"] is not None and abs(x["alpha"] - alpha) < 1e-12]
                    assert len(ref) == 1, (ds, r, alpha, rule, fb, len(ref))
                    ref, got = ref[0], res["rounds"][r][key(alpha, rule, fb)]
                    for g, v in ref["groups"].items():
                        a, b = v["threshold"], got["thresholds"][g]
                        assert (a is None and b is None) or (a is not None and b is not None and (float(a) == float(b) or abs(float(a) - float(b)) < 1e-9)), (ds, r, alpha, rule, g, a, b)
                    assert abs(ref["saving"] - got["saving"]) < 1e-9 and abs(ref["worst"] - got["worst"]) < 1e-12, (ds, r, alpha, rule, fb)
                    n += 1
    return n


def ci(k, n):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def stats(v):
    R = len(v)
    sv = np.array([x["saving"] for x in v]) * 100
    wv = np.array([x["worst"] for x in v]) * 100
    k = {c: int(sum(x[c] for x in v)) for c in ("exceed_test", "exceed_heldout", "exceed_pool")}
    return dict(rounds=R, saving_mean=float(sv.mean()), saving_sd=float(sv.std(ddof=1)), saving_p10=float(np.percentile(sv, 10)),
                saving_p90=float(np.percentile(sv, 90)), worst_mean=float(wv.mean()), worst_sd=float(wv.std(ddof=1)),
                coverage_mean=float(np.mean([x["coverage"] for x in v]) * 100),
                **k, **{f"{c}_ci": ci(k[c], R) for c in k})


def summarize(results):
    S, SA = {}, {}
    for ds, res in results.items():
        rounds = res["rounds"]
        S[ds] = {rule: stats([rd[key(MAIN_ALPHA[ds], rule, "resume")] for rd in rounds]) for rule in RULES}
        SA[ds] = {"always_cheap_saving": float(np.mean([rd["_cheap_saving"] for rd in rounds]) * 100)}
        for alpha in ALPHA_SET:
            SA[ds][f"{alpha:g}"] = {f"{rule}|{fb}": stats([rd[key(alpha, rule, fb)] for rd in rounds]) for rule, fb in RUNS}
    return S, SA


def report(S, SA, n_checked):
    L = ["# F6: 1,000 calibration/test rounds", "",
         f"Rounds 0-4 reproduce the rounds stored in F2/P2 ({n_checked} rule runs checked). Main configuration.", "",
         "- exceed, TEST sample: some type's rate on the TEST sample is above alpha.",
         "- exceed, held-out: some type's image-weighted rate on ALL TEST questions is above alpha.",
         "- exceed, pool: the same on all CAL and TEST images (includes CAL, optimistic).", ""]
    pc = lambda x, c: f"{100 * x[c] / x['rounds']:.1f}% [{100 * x[c + '_ci'][0]:.1f}, {100 * x[c + '_ci'][1]:.1f}]"
    for ds, rs in S.items():
        L += [f"## {ds}, main alpha {MAIN_ALPHA[ds]:.0%}, resume", "",
              "| rule | saving mean ± sd | worst mean ± sd | exceed, TEST sample | exceed, held-out | exceed, pool | coverage |",
              "|---|---|---|---|---|---|---|"]
        for rule, x in rs.items():
            L.append(f"| {NAME[rule]} | {x['saving_mean']:.1f} ± {x['saving_sd']:.1f} | {x['worst_mean']:.1f} ± {x['worst_sd']:.1f} | "
                     f"{pc(x, 'exceed_test')} | {pc(x, 'exceed_heldout')} | {pc(x, 'exceed_pool')} | {x['coverage_mean']:.1f}% |")
        L += ["", f"### {ds}, all alphas (always cheap saves {SA[ds]['always_cheap_saving']:.1f}%)", "",
              "| alpha | rule | saving mean ± sd [p10, p90] | worst mean ± sd | exceed, held-out | coverage |", "|---|---|---|---|---|---|"]
        for alpha in ALPHA_SET:
            for rk, x in SA[ds][f"{alpha:g}"].items():
                rule, fb = rk.split("|")
                L.append(f"| {alpha:.0%} | {NAME[rule]}, {fb} | {x['saving_mean']:.1f} ± {x['saving_sd']:.1f} [{x['saving_p10']:.1f}, "
                         f"{x['saving_p90']:.1f}] | {x['worst_mean']:.1f} ± {x['worst_sd']:.1f} | {pc(x, 'exceed_heldout')} | {x['coverage_mean']:.1f}% |")
        L.append("")
    return "\n".join(L) + "\n"


def write(results):
    n_checked = check_against_reports(results)
    S, SA = summarize(results)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    (OUT / "summary_alpha.json").write_text(json.dumps(SA, indent=1))
    (OUT / "report.md").write_text(report(S, SA, n_checked))
    print(report(S, SA, n_checked))


# ------------------------------------------------------------------------------------------------ fixed cuts (no calibration)
F4_DIR, P4_DIR, F7_DIR = RESULTS_DIR / "F4_reducers_test", RESULTS_DIR / "P4_pope_reducers_test", RESULTS_DIR / "F7_reducers_cal"


def fixed_answers(ds, spec):
    """Per question (array aligned with spec['D']): does the fixed cut's answer differ from the full answer? Always cheap
    uses the paired runs; VisionZip uses F4 / P4 (round-0 TEST) and F7 (round-0 CAL), together all non-DEV questions."""
    from utils.tools import read_jsonl
    D = spec["D"]
    out = {"always_cheap": D["y"].astype(bool)}            # y = cheap answer differs from the full answer (strict loss)
    if ds == "pope":
        rows = {r["qid"]: r for r in read_jsonl(RESULTS_DIR / "P1_pope_paired_caltest" / "traces.jsonl")}
        full = lambda q: rows[q]["full_answer"]
        vz = {**{r["qid"]: r for r in read_jsonl(P4_DIR / "pope_visionzip" / "traces.jsonl")},
              **{r["qid"]: r for r in read_jsonl(F7_DIR / "pope_visionzip" / "traces.jsonl")}}
    else:
        rows = {r["qid"]: r for r in read_jsonl(F2.F1 / ds / "traces.jsonl")}
        full = (lambda q: rows[q]["full_answer"]) if ds == "sugarcrepe" else (lambda q: rows[q]["full_norm"])
        vz = {**{r["qid"]: r for r in read_jsonl(F4_DIR / f"{ds}_visionzip" / "traces.jsonl")},
              **{r["qid"]: r for r in read_jsonl(F7_DIR / f"{ds}_visionzip" / "traces.jsonl")}}
    missing = [q for q in D["qid"] if q not in vz]
    assert not missing, f"{ds}: VisionZip answers missing for {len(missing)} questions (run exp_reducers_cal)"
    out["visionzip"] = np.array([vz[q]["answer"] != full(q) for q in D["qid"]])
    return out


def fixed_round(spec, changed, r, alpha):
    D, groups = spec["D"], spec["groups"]
    cal_imgs, test_imgs = split(spec, r)
    pos = {q: i for i, q in enumerate(D["qid"])}
    tg, _ = spec["test"](test_imgs, r)
    test_set, pool_set = set(test_imgs), set(test_imgs) | set(cal_imgs)
    is_test = np.array([im in test_set for im in D["img"]]); is_pool = np.array([im in pool_set for im in D["img"]])
    res = {}
    for name, y in changed.items():
        samp = {g: [y[pos[x["qid"]]] for x in tg if x["group"] == g] for g in groups}
        rates = {g: float(np.mean(v)) for g, v in samp.items() if v}

        def img_risk(mask, g):
            sel = np.flatnonzero(mask & (D["g"] == g))
            if not len(sel):
                return None
            per = collections.defaultdict(list)
            for i in sel:
                per[D["img"][i]].append(float(y[i]))
            return float(np.mean([np.mean(v) for v in per.values()]))
        held = {g: img_risk(is_test, g) for g in groups}
        pool = {g: img_risk(is_pool, g) for g in groups}
        res[name] = dict(worst=max(rates.values()), exceed_test=any(v > alpha for v in rates.values()),
                         exceed_heldout=any(v is not None and v > alpha for v in held.values()),
                         exceed_pool=any(v is not None and v > alpha for v in pool.values()), heldout=held)
    return res


def run_fixed(args):
    ds, R = args
    spec = build_spec(ds)
    changed = fixed_answers(ds, spec)
    alpha = MAIN_ALPHA[ds]
    rounds = [fixed_round(spec, changed, r, alpha) for r in range(R)]
    S = {}
    for name in changed:
        v = [rd[name] for rd in rounds]
        wv = np.array([x["worst"] for x in v]) * 100
        k = {c: int(sum(x[c] for x in v)) for c in ("exceed_test", "exceed_heldout", "exceed_pool")}
        S[name] = dict(rounds=R, worst_mean=float(wv.mean()), worst_sd=float(wv.std(ddof=1)), **k,
                       **{f"{c}_ci": ci(k[c], R) for c in k})
    print(f"{ds}: fixed cuts over {R} rounds", flush=True)
    return ds, S


def write_fixed(R):
    """Fixed cuts over the same rounds, main alpha. Saving is not replayed: the cuts use no calibration, and their saving is
    the exclusive round-0 measurement of F4 / P4 (summary_fixed.json records it next to the counts)."""
    with Pool(3) as p:
        S = dict(p.map(run_fixed, [(ds, R) for ds in ("sugarcrepe", "textvqa", "pope")]))
    f4 = json.loads((F4_DIR / "summary.json").read_text())
    p4 = json.loads((P4_DIR / "summary.json").read_text())
    T = json.loads((F2.OUT / "test_results.json").read_text())
    TP = json.loads((RESULTS_DIR / "P2_pope_certify_evaluate" / "test_results_pope.json").read_text())
    for ds in S:
        S[ds]["visionzip"]["saving_round0"] = 100 * (p4["visionzip"]["saving"] if ds == "pope" else f4[f"{ds}_visionzip"]["saving"])
        runs = TP["rounds"][0]["runs"] if ds == "pope" else T["datasets"][ds]["rounds"][0]["runs"]
        cheap = [x for x in runs if x["rule"] == "always_cheap"][0]
        S[ds]["always_cheap"]["saving_round0"] = 100 * cheap["saving"]
    (OUT / "summary_fixed.json").write_text(json.dumps(S, indent=1))
    L = ["", "## Fixed cuts (no calibration), main alpha", "",
         "| dataset | cut | saving, round 0 (%) | worst mean ± sd | exceed, TEST sample | exceed, held-out | exceed, pool |", "|---|---|---|---|---|---|---|"]
    pc = lambda x, c: f"{100 * x[c] / x['rounds']:.1f}%"
    for ds, v in S.items():
        for name, x in v.items():
            L.append(f"| {ds} | {name} | {x['saving_round0']:.1f} | {x['worst_mean']:.1f} ± {x['worst_sd']:.1f} | {pc(x, 'exceed_test')} | "
                     f"{pc(x, 'exceed_heldout')} | {pc(x, 'exceed_pool')} |")
    rep = OUT / "report.md"
    txt = rep.read_text().split("\n## Fixed cuts")[0].rstrip("\n") + "\n"
    rep.write_text(txt + "\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--summarize", action="store_true", help="only rebuild the summaries and report from rounds.json.gz")
    ap.add_argument("--fixed", action="store_true", help="only the fixed cuts (always cheap, VisionZip); needs F7_reducers_cal")
    a = ap.parse_args()
    if a.fixed:
        write_fixed(a.rounds)
        return
    if a.summarize:
        write(json.loads(gzip.decompress((OUT / "rounds.json.gz").read_bytes())))
        return
    with Pool(3) as p:
        results = dict(p.map(run_dataset, [(ds, a.rounds) for ds in ("sugarcrepe", "textvqa", "pope")]))
    OUT.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(results, default=lambda o: o.item() if isinstance(o, np.generic) else o).encode()
    (OUT / "rounds.json.gz").write_bytes(gzip.compress(blob))
    old = OUT / "rounds.json"
    if old.exists():
        old.unlink()                         # superseded by rounds.json.gz (all alphas)
    write(results)


if __name__ == "__main__":
    main()
