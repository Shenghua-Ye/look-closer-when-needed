"""POPE protocol: DEV report, then the POPE lock. No model runs; no CAL or TEST model output is read.

    python -m exp.exp_pope_protocol --dev_report    # DEV paired runs: change rates, AUROC, forced vs free answers, paths
    python -m exp.exp_pope_protocol --lock          # writes results/P0_pope_protocol_lock/ once (refuses to change it)

What the lock fixes before CAL (same rules as the main study's F0 lock and its TEST supplement, for dataset code 3):
  - groups: size terciles (manifest rule pope-size-tercile-v1; the 79-row table is in pope_meta.json)
  - five rounds: round 0 = manifest split; rounds 1-4 keep DEV and re-split the other images, CAL if u < 0.5 with
    u = default_rng([20261101, 3, r])
  - CAL and TEST samples: one question per (image, group) and one per image, seeded draws over sorted keys
    (CAL default_rng([20261102, 3, r, 0|1]), TEST default_rng([20261103, 3, r, 0|1]))
  - eligibility per round and alpha: n_g >= nmin(alpha, delta / |G|)
  - candidate threshold paths per group and shared ('*'), built with exp_frontier.pareto_path (always_probe, restart cost
    model, the main study's rule) on the DEV pilot band only (u < 0.3); configurations: forced-choice margin with the
    strict loss (main) and with the accuracy-drop loss (variant)
  - main certificate: per-type LTT with the resume fallback at alpha = 5%; the other rules and the restart fallback
    are reported separately; alphas 5/10/20% as sensitivity
"""
import argparse
import collections
import json
import platform

import numpy as np

from exp import exp_frontier as E6
from exp import pope_common as C

N_ROUNDS = 5
CONFIGS = [("signal-first_margin_loss-strict", "strict"), ("signal-first_margin_loss-acc_drop", "acc_drop")]


def _num(x):
    return float(x) if np.isfinite(x) else ("-inf" if x < 0 else "inf")


def dev_rows(band=None):
    rows = C.read_jsonl(C.DEV_DIR / "traces.jsonl")
    return [r for r in rows if band is None or r["dev_band"] == band]


def e6_arrays(rows, loss):
    """The arrays E6.candidate_stats / pareto_path read (SugarCrepe-style cost model: no decode, restart fallback)."""
    f = lambda k: np.array([r[k] for r in rows], float)
    acc_c = np.array([r["base_answer"] == r["correct"] for r in rows], float)
    acc_f = np.array([r["full_answer"] == r["correct"] for r in rows], float)
    changed = np.array([bool(r["changed"]) for r in rows])
    P, M = f("ms_P"), f("ms_M")
    return dict(n=len(rows), img=np.array([r["image_id"] for r in rows]), g=np.array([r["group"] for r in rows]),
                y=changed if loss == "strict" else acc_c < acc_f, m=f("base_margin"), S0=f("ms_S_pre") + f("ms_S_enc"),
                Gc=np.zeros(len(rows)), P=P, M=M, C=P + M, H=f("ms_H_pre") + f("ms_H_fwd") + f("ms_H_score"))


def candidate_paths(loss):
    rows = dev_rows("pilot")
    D = e6_arrays(rows, loss)
    out = {}
    for g in C.GROUPS + ["*"]:
        idx = np.arange(D["n"]) if g == "*" else np.flatnonzero(D["g"] == g)
        cands, risk, cost = E6.candidate_stats(D, idx, "always_probe", None)
        stats = dict(zip(cands, zip(risk, cost)))
        path = E6.pareto_path(D, idx, "always_probe", None)
        out[g] = dict(n_dev_questions=len(idx), n_dev_images=len(np.unique(D["img"][idx])), dev_full_ms=round(float(D["H"][idx].mean()), 2),
                      path=[dict(lam=_num(th[0]), dev_risk=round(float(stats[th][0]), 5), dev_cost_ms=round(float(stats[th][1]), 2)) for th in path])
    return out


def rounds():
    imgs = sorted(C.images().values(), key=lambda x: x["image_id"])
    ids = lambda s: [x["image_id"] for x in imgs if x["split"] == s]
    dev, rest = ids("DEV"), [x["image_id"] for x in imgs if x["split"] != "DEV"]
    out = [dict(CAL=ids("CAL"), TEST=ids("TEST"))]
    for r in range(1, N_ROUNDS):
        u = np.random.default_rng([C.SEEDS["rounds"], C.CODE, r]).random(len(rest))
        out.append(dict(CAL=[i for i, v in zip(rest, u) if v < 0.5], TEST=[i for i, v in zip(rest, u) if v >= 0.5]))
    for sp in out:
        assert not set(sp["CAL"]) & set(sp["TEST"]) and set(sp["CAL"]) | set(sp["TEST"]) == set(rest)
    return dev, out


def sample(img_ids, r, seed):
    keep = set(img_ids)
    by_g, by_img = collections.defaultdict(list), collections.defaultdict(list)
    for q in C.questions():
        if q["image_id"] in keep:
            by_g[(q["image_id"], q[C.GROUP_KEY])].append(q["qid"])
            by_img[q["image_id"]].append(q["qid"])
    rng_g = np.random.default_rng([seed, C.CODE, r, 0])
    rng_i = np.random.default_rng([seed, C.CODE, r, 1])
    per_group = [dict(image_id=i, group=g, qid=sorted(v)[rng_g.integers(len(v))]) for (i, g), v in sorted(by_g.items())]
    per_image = [dict(image_id=i, qid=sorted(v)[rng_i.integers(len(v))]) for i, v in sorted(by_img.items())]
    return per_group, per_image


def eligibility(per_group, n_img):
    out = {}
    n = collections.Counter(x["group"] for x in per_group)
    for alpha in E6.ALPHAS:
        need = E6.nmin(alpha, E6.DELTA / len(C.GROUPS))
        elig = [g for g in C.GROUPS if n[g] >= need]
        out[f"{alpha:g}"] = dict(n_cal={g: n[g] for g in C.GROUPS}, n_min=need, eligible=elig,
                                 level=E6.DELTA / max(1, len(elig)), overall_ltt_eligible=n_img >= E6.nmin(alpha, E6.DELTA))
    return out


def dev_report():
    from sklearn.metrics import roc_auc_score
    rows = dev_rows()
    L = ["# POPE DEV report", "", f"DEV paired runs: {len(rows)} questions on {len({r['image_id'] for r in rows})} images "
         "(pilot band builds the paths; rehearsal band is reserved for the code rehearsal).", ""]
    for band in ("pilot", "rehearsal", None):
        rs = [r for r in rows if band is None or r["dev_band"] == band]
        y = np.array([r["changed"] for r in rs])
        H = np.array([r["ms_H_pre"] + r["ms_H_fwd"] + r["ms_H_score"] for r in rs])
        Cc = np.array([r["ms_S_pre"] + r["ms_S_enc"] + r["ms_P"] + r["ms_M"] for r in rs])
        Hr = np.array([r["ms_R_pre"] + r["ms_R_vis"] + r["ms_R_cont"] + r["ms_R_dec"] for r in rs])
        auc = roc_auc_score(y, -np.array([r["base_margin"] for r in rs])) if 0 < y.sum() < len(y) else float("nan")
        L += [f"## {band or 'all DEV'} ({len(rs)} questions)", "",
              f"- Changed (cheap != full): {y.mean():.1%}; resume != full: {np.mean([r['resume_ne_full'] for r in rs]):.2%} "
              f"({sum(r['resume_ne_full'] for r in rs)}); AUROC of the forced-choice margin for a change: {auc:.3f}",
              f"- Accuracy: cheap {np.mean([r['base_answer'] == r['correct'] for r in rs]):.1%}, full {np.mean([r['full_answer'] == r['correct'] for r in rs]):.1%}; "
              f"yes-ratio cheap {np.mean([r['base_answer'] == 'yes' for r in rs]):.1%}, full {np.mean([r['full_answer'] == 'yes' for r in rs]):.1%} "
              f"(gold {np.mean([r['correct'] == 'yes' for r in rs]):.1%})",
              f"- Forced choice vs free greedy answer of the full pass: agree {np.mean([r['full_gen_agrees'] for r in rs]):.2%}; "
              f"free answer not yes/no: {np.mean([r['full_gen_word'] not in ('yes', 'no') for r in rs]):.2%}",
              f"- Mean ms: full {H.mean():.1f}, cheap {Cc.mean():.1f} ({1 - Cc.mean() / H.mean():.1%} saved if always cheap), resume continuation {Hr.mean():.1f}",
              f"- Cache checks failed: {sum(r['kv_injected'] != 1 or r['kv_len'] != r['kv_expected'] for r in rs)}", ""]
        L += ["| type | questions | images | changed | AUROC |", "|---|---|---|---|---|"]
        for g in C.GROUPS:
            gs = [r for r in rs if r["group"] == g]
            yy = np.array([r["changed"] for r in gs])
            a = roc_auc_score(yy, -np.array([r["base_margin"] for r in gs])) if 0 < yy.sum() < len(yy) else float("nan")
            L.append(f"| {g} | {len(gs)} | {len({r['image_id'] for r in gs})} | {yy.mean():.1%} ({yy.sum()}) | {a:.3f} |")
        L.append("")
    for name, loss in CONFIGS:
        paths = candidate_paths(loss)
        L += [f"## Candidate paths (pilot band), {name}", "", "| group | DEV questions / images | full ms | steps | path (lambda: risk, ms) |", "|---|---|---|---|---|"]
        for g, p in paths.items():
            steps = "; ".join(f"{s['lam'] if isinstance(s['lam'], str) else round(s['lam'], 3)}: {s['dev_risk']:.3f}, {s['dev_cost_ms']:.0f}" for s in p["path"])
            L.append(f"| {g} | {p['n_dev_questions']} / {p['n_dev_images']} | {p['dev_full_ms']:.0f} | {len(p['path'])} | {steps} |")
        L.append("")
    dev, rds = rounds()
    L += ["## Eligibility per round (counts only; no CAL output read)", "", "| round | CAL images | TEST images | " +
          " | ".join(f"alpha {a:.0%}: n per type (need)" for a in E6.ALPHAS) + " |", "|---|---|---|" + "---|" * len(E6.ALPHAS)]
    for r, sp in enumerate(rds):
        pg, pi = sample(sp["CAL"], r, C.SEEDS["cal_sample"])
        el = eligibility(pg, len(pi))
        L.append(f"| {r} | {len(sp['CAL'])} | {len(sp['TEST'])} | " + " | ".join(
            f"{', '.join(str(el[f'{a:g}']['n_cal'][g]) for g in C.GROUPS)} ({el[f'{a:g}']['n_min']})" for a in E6.ALPHAS) + " |")
    (C.DEV_DIR / "dev_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def build_lock():
    dev, rds = rounds()
    files, rec = {}, []
    cal_s, test_s = [], []
    for r, sp in enumerate(rds):
        pg, pi = sample(sp["CAL"], r, C.SEEDS["cal_sample"])
        tg, ti = sample(sp["TEST"], r, C.SEEDS["test_sample"])
        cal_s.append(dict(round=r, per_group=pg, per_image=pi))
        test_s.append(dict(round=r, per_group=tg, per_image=ti))
        rec.append(dict(round=r, n_cal_images=len(sp["CAL"]), n_test_images=len(sp["TEST"]), eligibility=eligibility(pg, len(pi))))
    files["splits_pope.json"] = json.dumps(dict(DEV=dev, rounds=rds))
    files["cal_sample_pope.json"] = json.dumps(cal_s)
    files["test_sample_pope.json"] = json.dumps(test_s)
    meta = json.loads((C.MAN / "pope_meta.json").read_text())
    lock = dict(
        dataset="pope", groups=C.GROUPS, group_rule=meta["rules_version"], size_table=meta["size_table"],
        unit=meta["unit"], answer="forced choice yes/no at the answer position", yes_tokens=list(C.YES), no_tokens=list(C.NO),
        main=dict(alpha=C.ALPHA_MAIN, signal="first_margin (forced-choice margin)", loss="strict", rule="group_ltt",
                  fallback="resume", alphas_sensitivity=list(E6.ALPHAS)),
        testing=dict(delta=E6.DELTA, eligibility="n_g >= nmin(alpha, delta / |G|)", within_group="fixed sequence along the candidate path",
                     across_groups="Bonferroni over eligible groups", rules="exp_f2_certify.RULES, each with restart and resume",
                     rounds="round 0 is the main result; rounds 1-4 describe split sensitivity"),
        seeds=C.SEEDS, rounds=rec,
        paths={name: candidate_paths(loss) for name, loss in CONFIGS},
        python=platform.python_version(), numpy=np.__version__)
    return lock, files


def write_lock():
    C.LOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = C.LOCK_DIR / "pope_lock.json"
    lock, files = build_lock()
    lock = json.loads(json.dumps(lock))
    if path.exists():
        old = json.loads(path.read_text())
        diff = sorted(k for k in set(old) | set(lock) if old.get(k) != lock.get(k))
        if diff:
            raise SystemExit(f"{path} holds a different lock (differs in: {', '.join(diff)}); the lock is written once")
        print("POPE lock unchanged")
        return
    for f, text in files.items():
        (C.LOCK_DIR / f).write_text(text)
    path.write_text(json.dumps(lock, indent=1))
    print("POPE lock written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_report", action="store_true")
    ap.add_argument("--lock", action="store_true")
    a = ap.parse_args()
    if a.dev_report:
        dev_report()
    if a.lock:
        write_lock()


if __name__ == "__main__":
    main()
