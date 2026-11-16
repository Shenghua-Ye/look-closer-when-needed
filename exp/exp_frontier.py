"""Loading the DEV paired runs, the cost model and the candidate thresholds (used by exp_protocol, exp_pope_protocol,
exp_f2_certify and exp_rounds_extra). No model runs.

Cost of a request, from the stage times recorded in the paired runs:
  full pass (always full, or a type without a threshold)   H
  accepted cheap answer                                     S0 + C       (S0: base-view encode; C: prefill + answer)
  fallback with restart                                     S0 + P + M + H
where C = P + M + L (prefill, answer scoring, decoding). For confidence signals read after the whole cheap answer
(seq_*), a fallback also pays for the cheap decode, so L moves into P.

Candidate thresholds of a type: 21 evenly spaced quantiles of its confidences on DEV plus -inf and +inf. pareto_path
keeps, in order from strict to loose, each candidate that is faster than every stricter one and faster than always
using the full pass; the order is fixed before CAL is used.
"""
import json

import numpy as np

from root import E2_DIR, E5_DIR, RESULTS_DIR

ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.10
SIGNAL_DIR = RESULTS_DIR / "E8_textvqa_signals_dev"
SPECS = {
    "sugarcrepe": dict(src=E2_DIR, group="obs_group", groups=["replace", "add_remove", "reorder", "other"]),
    "textvqa": dict(src=E5_DIR, group="qtype", groups=["reading", "counting", "existence", "attribute", "relation", "other"]),
}
SAFE, CHEAP, ACCEPT, FALLBACK = 4, 0, 2, 3  # action codes


# ------------------------------------------------------------------------------------------------ data
def load(dataset, exp_dir, limit=0, loss="strict", signal="first_margin"):
    """The DEV paired run of a dataset as arrays: type, loss, confidence and stage times per question.
    loss: strict (answer differs from the full pass) or acc_drop (less accurate than the full pass; needs labels).
    signal: first_margin, or a TextVQA signal from results/E8_textvqa_signals_dev/ (e.g. seq_min_margin)."""
    spec = SPECS[dataset]
    rows = [json.loads(l) for l in open(exp_dir / "traces.jsonl")]
    if limit:
        rows = [r for r in rows if r["order"] < limit]
    f = lambda k: np.array([r[k] for r in rows], dtype=float)
    if dataset == "sugarcrepe":
        acc_c = np.array([r["base_answer"] == r["correct"] for r in rows], float)
        acc_f = np.array([r["full_answer"] == r["correct"] for r in rows], float)
    else:
        acc_c, acc_f = f("base_acc"), f("full_acc")
    changed = np.array([bool(r["changed"]) for r in rows])
    D = dict(n=len(rows), img=np.array([r["image_id"] for r in rows]), g=np.array([r[spec["group"]] for r in rows]),
             y=changed if loss == "strict" else acc_c < acc_f, acc_c=acc_c, acc_f=acc_f, m=f("base_margin"),
             S0=f("ms_S_pre") + f("ms_S_enc"), P=f("ms_P"))
    if dataset == "sugarcrepe":
        D.update(H=f("ms_H_pre") + f("ms_H_fwd") + f("ms_H_score"), M=f("ms_M"), L=np.zeros(len(rows)))
    else:
        D.update(H=f("ms_H"), M=np.zeros(len(rows)), L=f("ms_D"))
    if signal != "first_margin":
        sig = {r["qid"]: r for r in map(json.loads, open(SIGNAL_DIR / "signals.jsonl"))}
        D["m"] = np.array([sig[r["qid"]][signal] for r in rows], dtype=float)
        if signal.startswith("seq_"):  # decided after the whole cheap answer: a fallback also wastes the cheap decode
            D["P"], D["L"] = D["P"] + D["L"], np.zeros(len(rows))
    D["C"] = D["P"] + D["M"] + D["L"]  # cheap answer after the base encode
    return D


# ------------------------------------------------------------------------------------------------ policies
def outcome(D, idx, policy, th, s=None):
    """Cost (ms), loss (a returned answer that differs from full) and action per request; th None = always-full path.
    policy: always_probe (accept the cheap answer if m > th[0], else fall back) or group_fixed (always cheap)."""
    H, S0, C, P, M, y, m = (D[k][idx] for k in ("H", "S0", "C", "P", "M", "y", "m"))
    if th is None:
        return H, np.zeros(len(idx), bool), np.full(len(idx), SAFE)
    if policy == "group_fixed":
        return S0 + C, y.copy(), np.full(len(idx), CHEAP)
    if policy == "always_probe":
        acc = m > th[0]
        return S0 + np.where(acc, C, P + M + H), y & acc, np.where(acc, ACCEPT, FALLBACK)
    raise ValueError(policy)


def _grid(v, k):
    return np.unique(np.r_[-np.inf, np.quantile(v, np.linspace(0, 1, k)), np.inf])


def candidate_stats(D, idx, policy, s=None):
    """All candidate settings of a policy with their mean risk and cost on idx (vectorised over thresholds)."""
    H, S0, C, P, M, y, m = (D[k][idx] for k in ("H", "S0", "C", "P", "M", "y", "m"))
    if policy == "group_fixed":
        return [("cheap",)], np.array([np.mean(y)]), np.array([np.mean(S0 + C)])
    if policy == "always_probe":
        lam = _grid(m, 21)
        acc = m[:, None] > lam[None, :]
        cost = (S0[:, None] + np.where(acc, C[:, None], (P + M + H)[:, None])).mean(0)
        return [(l,) for l in lam], (y[:, None] & acc).mean(0), cost
    raise ValueError(policy)


def pareto_path(D, idx, policy, s=None):
    """Settings ordered from the safest to the cheapest, each cheaper than everything safer, starting below the cost of
    the always-full path."""
    cands, risk, cost = candidate_stats(D, idx, policy, s)
    path, best = [], D["H"][idx].mean()
    for j in np.lexsort((cost, risk)):
        if cost[j] < best - 1e-9:
            path.append(cands[j]); best = cost[j]
    return path


# ------------------------------------------------------------------------------------------------ calibration
def nmin(alpha, level):
    """Fewest independent samples for which zero observed changes certify R <= alpha at this level."""
    return int(np.ceil(np.log(level) / np.log(1 - alpha)))
