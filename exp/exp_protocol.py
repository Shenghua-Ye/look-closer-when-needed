"""F0: lock the protocol before the formal CAL / TEST runs. No model runs; no CAL or TEST model output is read.

    python run.py --task_name protocol_lock --use_gpu False      # writes results/F0_protocol_lock/

What is fixed before calibration, for both datasets:
  - Candidate threshold paths, one per group plus the shared path "*" (shared-threshold and overall LTT), built with
    exp_frontier.pareto_path on the DEV paired runs (pilot_dev). Main configuration and the reported variants.
    All five rounds use these same paths.
  - Five CAL / TEST rounds. Round 0 is the frozen manifest split. Rounds 1-4 keep DEV fixed and re-split only the
    images never used for development: each non-DEV image is CAL or TEST with probability 1/2, from a fresh u per round
    (the same rule as the manifest split, CAL 0.4 <= u < 0.7, TEST u >= 0.7). Proportions stay 40/30/30.
  - The CAL sample of every round, drawn answer-blind: one question per (image, group) for the per-group tests and one
    per image for overall LTT.
  - Eligibility per round and alpha, from CAL counts only: n_g >= nmin(alpha, delta / |G|); test level
    delta / |G_eligible|.
protocol_lock.json is written once. A rerun that would change it stops; delete the folder to rebuild it.
"""
import collections
import json
import platform

import numpy as np

from data_provider import datasets
from exp import exp_frontier as E6
from exp.exp_basic import ExpBasic
from root import E2_DIR, RESULTS_DIR, results_dir
from utils.tools import read_jsonl

OUT_DIR = RESULTS_DIR / "F0_protocol_lock"
ROUND_SEED, CAL_SEED = 20261101, 20261102
N_ROUNDS = 5
CODE = {"sugarcrepe": 1, "textvqa": 2}  # dataset index in the seed streams
MAIN = {"sugarcrepe": dict(alpha=0.05, signal="first_margin", loss="strict"),   # first_margin = A/B margin at the answer
        "textvqa": dict(alpha=0.10, signal="seq_min_margin", loss="strict")}
VARIANTS = {"sugarcrepe": [("first_margin", "acc_drop")],                         # score-drop target (supplementary)
            "textvqa": [("first_margin", "strict"), ("seq_min_margin", "acc_drop")]}  # first-token ablation; score drop


def _num(x):
    return float(x) if np.isfinite(x) else ("-inf" if x < 0 else "inf")


def rounds(name):
    """DEV image ids and, per round, the CAL and TEST image ids."""
    imgs = sorted(read_jsonl(datasets[name]["images"]), key=lambda x: x["image_id"])
    ids = lambda s: [x["image_id"] for x in imgs if x["split"] == s]
    dev, rest = ids("DEV"), [x["image_id"] for x in imgs if x["split"] != "DEV"]
    out = [dict(CAL=ids("CAL"), TEST=ids("TEST"))]
    for r in range(1, N_ROUNDS):
        u = np.random.default_rng([ROUND_SEED, CODE[name], r]).random(len(rest))
        out.append(dict(CAL=[i for i, v in zip(rest, u) if v < 0.5], TEST=[i for i, v in zip(rest, u) if v >= 0.5]))
    for sp in out:
        assert not set(sp["CAL"]) & set(sp["TEST"]) and set(sp["CAL"]) | set(sp["TEST"]) == set(rest)
    return dev, out


def cal_sample(name, cal_imgs, r):
    """Answer-blind CAL sample: one question per (image, group) and one per image (sorted keys, seeded draws)."""
    gk, cal = datasets[name]["group_key"], set(cal_imgs)
    by_g, by_img = collections.defaultdict(list), collections.defaultdict(list)
    for q in read_jsonl(datasets[name]["questions"]):
        if q["image_id"] in cal:
            by_g[(q["image_id"], q[gk])].append(q["qid"])
            by_img[q["image_id"]].append(q["qid"])
    rng_g = np.random.default_rng([CAL_SEED, CODE[name], r, 0])
    rng_i = np.random.default_rng([CAL_SEED, CODE[name], r, 1])
    per_group = [dict(image_id=i, group=g, qid=sorted(v)[rng_g.integers(len(v))]) for (i, g), v in sorted(by_g.items())]
    per_image = [dict(image_id=i, qid=sorted(v)[rng_i.integers(len(v))]) for i, v in sorted(by_img.items())]
    return per_group, per_image


def eligibility(groups, per_group, n_img):
    out = {}
    for alpha in E6.ALPHAS:
        n = collections.Counter(x["group"] for x in per_group)
        need = E6.nmin(alpha, E6.DELTA / len(groups))
        elig = [g for g in groups if n[g] >= need]
        out[f"{alpha:g}"] = dict(n_cal={g: n[g] for g in groups}, n_min=need, eligible=elig,
                                 level=E6.DELTA / max(1, len(elig)), overall_ltt_eligible=n_img >= E6.nmin(alpha, E6.DELTA))
    return out


def candidate_paths(name, signal, loss):
    """Per-group and shared paths of the margin-only policy on the DEV paired runs, safest first."""
    spec = E6.SPECS[name]
    D = E6.load(name, spec["src"], loss=loss, signal=signal)
    out = {}
    for g in spec["groups"] + ["*"]:
        idx = np.arange(D["n"]) if g == "*" else np.flatnonzero(D["g"] == g)
        cands, risk, cost = E6.candidate_stats(D, idx, "always_probe", None)
        stats = dict(zip(cands, zip(risk, cost)))
        path = E6.pareto_path(D, idx, "always_probe", None)
        out[g] = dict(n_dev_questions=len(idx), n_dev_images=len(np.unique(D["img"][idx])),
                      dev_full_ms=round(float(D["H"][idx].mean()), 2),
                      path=[dict(lam=_num(th[0]), dev_risk=round(float(stats[th][0]), 5), dev_cost_ms=round(float(stats[th][1]), 2))
                            for th in path])
    return out


def build_lock():
    """The lock and its side files ({file name: text}); nothing is written here."""
    files = {}
    lock = dict(
        model={k: json.loads((E2_DIR / "run_config.json").read_text())[k]
               for k in ("model", "revision", "dtype", "attn_implementation", "batch_size", "torch", "transformers")},
        cheap_path="base view only (336 px, 577 visual tokens); fallback reuses nothing (rho = 0)",
        loss="strict: returned answer != full answer (TextVQA after vqa-norm-v1); fallback and direct-full losses are 0",
        main={k: dict(v, alphas_sensitivity=list(E6.ALPHAS)) for k, v in MAIN.items()},
        signal_rules=dict(sugarcrepe="|log p_A - log p_B| at the answer position (token ids from run_config)",
                          textvqa="min over generated steps of p_top1 - p_top2: EOS step counted, all 32 steps if truncated, "
                                  "first-token ties by softmax top-1 and later tokens by fp16 logits argmax "
                                  "(exp_signals.decode_with_signals)"),
        groups={name: E6.SPECS[name]["groups"] for name in datasets},
        group_rules={name: json.loads(datasets[name]["meta"].read_text())["rules_version"] for name in datasets},
        testing=dict(delta=E6.DELTA, per="dataset and round",
                     eligibility="n_g >= nmin(alpha, delta / |G|), the fewest samples with which zero changes pass; counts only",
                     within_group="fixed sequence along the candidate path from the safest setting; p = P(Bin(n_g, alpha) <= k_g); "
                                  "stop at the first p > delta / |G_eligible|; keep the last passing setting; none -> full path",
                     across_groups="Bonferroni over the eligible groups",
                     certificate="main policy only; baselines and variants are calibrated and reported separately",
                     rounds="round 0 is the main result; rounds 1-4 describe split sensitivity (mean, std); delta = 0.1 per "
                            "round, not a joint guarantee over the five (that would need delta = 0.02 per round)"),
        seeds=dict(manifest_split=20261001, rounds=ROUND_SEED, cal_sample=CAL_SEED,
                   streams="rounds: default_rng([rounds, dataset, r]); CAL: default_rng([cal_sample, dataset, r, 0 per group | 1 per image]); "
                           "dataset sugarcrepe = 1, textvqa = 2"),
        datasets={}, versions={})
    for name in datasets:
        dev, rds = rounds(name)
        groups = E6.SPECS[name]["groups"]
        splits = dict(DEV=dev, rounds=rds)
        samples, rec = [], []
        for r, sp in enumerate(rds):
            per_group, per_image = cal_sample(name, sp["CAL"], r)
            samples.append(dict(round=r, per_group=per_group, per_image=per_image))
            rec.append(dict(round=r, n_cal_images=len(sp["CAL"]), n_test_images=len(sp["TEST"]),
                            n_cal_per_group=len(per_group), n_cal_per_image=len(per_image),
                            eligibility=eligibility(groups, per_group, len(per_image))))
        files[f"splits_{name}.json"] = json.dumps(splits)
        files[f"cal_sample_{name}.json"] = json.dumps(samples)
        configs = [(MAIN[name]["signal"], MAIN[name]["loss"])] + VARIANTS[name]
        lock["datasets"][name] = dict(
            n_dev_images=len(dev), rounds=rec,
            paths={f"signal-{s}_loss-{l}": candidate_paths(name, s, l) for s, l in configs})
    lock["versions"] = dict(python=platform.python_version(), numpy=np.__version__)
    return lock, files


def write_report(out_dir, lock):
    L = ["# F0 protocol lock", "",
         "Fixed before any CAL or TEST model output was read. Machine-readable: `protocol_lock.json` (image lists in "
         "`splits_*.json`, CAL samples in `cal_sample_*.json`).", "",
         f"- Model: {lock['model']['model']} @ {lock['model']['revision'][:8]}, {lock['model']['dtype']}, "
         f"{lock['model']['attn_implementation']}, batch {lock['model']['batch_size']}; torch {lock['model']['torch']}, "
         f"transformers {lock['model']['transformers']}.",
         f"- Main configuration: SugarCrepe A/B margin, alpha = 5%; TextVQA min token margin, alpha = 10%; strict loss; "
         f"sensitivity at alpha = 5/10/20%. delta = {E6.DELTA} per dataset and round.",
         f"- Rounds: {lock['testing']['rounds']}.",
         f"- Seeds: manifest split {lock['seeds']['manifest_split']}, rounds {ROUND_SEED}, CAL sample {CAL_SEED} ({lock['seeds']['streams']}).", ""]
    for name, d in lock["datasets"].items():
        groups = lock["groups"][name]
        L += [f"## {name}", "", f"DEV images: {d['n_dev_images']} (fixed in every round; candidate paths use the pilot_dev "
              "paired runs).", "",
              "| Round | CAL images | TEST images | CAL samples per (image, group) | " +
              " | ".join(f"alpha {a:.0%}: eligible groups (level)" for a in E6.ALPHAS) + " |",
              "|---|---|---|---|" + "---|" * len(E6.ALPHAS)]
        for r in d["rounds"]:
            el = [r["eligibility"][f"{a:g}"] for a in E6.ALPHAS]
            L.append(f"| {r['round']} | {r['n_cal_images']} | {r['n_test_images']} | {r['n_cal_per_group']} | " +
                     " | ".join(f"{', '.join(e['eligible']) or 'none'} ({e['level']:.4f})" for e in el) + " |")
        e0 = d["rounds"][0]["eligibility"]
        L += ["", "CAL samples per group, round 0: " + ", ".join(f"{g} {e0['0.05']['n_cal'][g]}" for g in groups) +
              "; needed for eligibility: " + ", ".join(f"alpha {a:.0%}: {e0[f'{a:g}']['n_min']}" for a in E6.ALPHAS) + ".", ""]
        for cfg, P in d["paths"].items():
            L += [f"### Candidate paths, {cfg}", "",
                  "Settings in test order (safest first): threshold lambda, DEV change rate (all questions), DEV mean cost (ms).", "",
                  "| Group | DEV questions / images | full path ms | steps | path (lambda: risk, ms) |", "|---|---|---|---|---|"]
            for g, p in P.items():
                steps = "; ".join(f"{s['lam'] if isinstance(s['lam'], str) else round(s['lam'], 4)}: {s['dev_risk']:.3f}, {s['dev_cost_ms']:.0f}"
                                  for s in p["path"])
                L.append(f"| {g} | {p['n_dev_questions']} / {p['n_dev_images']} | {p['dev_full_ms']:.0f} | {len(p['path'])} | {steps} |")
            L.append("")
    (out_dir / "protocol_lock.md").write_text("\n".join(L) + "\n")


def run_lock(exp_dir=""):
    out_dir = results_dir(exp_dir, OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock, files = build_lock()
    lock = json.loads(json.dumps(lock))
    path = out_dir / "protocol_lock.json"
    if path.exists():
        old = json.loads(path.read_text())
        diff = sorted(k for k in set(old) | set(lock) if old.get(k) != lock.get(k))
        if diff:
            raise SystemExit(f"{path} holds a different lock (differs in: {', '.join(diff)}). The lock is written once; "
                             "delete the folder to rebuild it.")
        print(f"{path} unchanged")
    else:
        for f, text in files.items():
            (out_dir / f).write_text(text)
        path.write_text(json.dumps(lock, indent=1))
    write_report(out_dir, lock)
    print(f"protocol lock: {out_dir}")


class ExpProtocol(ExpBasic):
    def run(self):
        run_lock(self.args.exp_dir)
