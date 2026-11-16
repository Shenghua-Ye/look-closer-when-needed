"""Helpers shared by the paired runs and the online timing runs (exp_f1_paired, exp_f3_timing, exp_pope_*).

  KeepKV / keep_kv(P)  keep the KV cache of the cheap prefill, so that a fallback can continue from it
  warmup_set           warm-up requests: the first n plus one request of every AnyRes grid not yet covered
  gpu_state / exclusivity / own_gpu
                       read-only nvidia-smi checks that no other process uses a GPU while latency is measured
  _stats               mean / p50 / p95 / max of a list of latencies
"""
import contextlib
import os
import subprocess
from datetime import datetime

import numpy as np

UTIL_BUSY = 5  # % utilisation that marks a GPU as in use


# ------------------------------------------------------------------------------------------------ cache capture
class KeepKV:
    """Stands in for P.lm (on the instance) while the cheap path runs. Forwards every call unchanged, except that the
    one call asking for a cache without passing one (the cheap prefill) gets a DynamicCache that this object keeps."""

    def __init__(self, lm):
        from transformers import DynamicCache
        self.lm, self.new_cache, self.cache, self.n_injected = lm, DynamicCache, None, 0

    def __call__(self, *args, **kwargs):
        if kwargs.get("use_cache") and kwargs.get("past_key_values") is None:
            kwargs["past_key_values"] = self.cache = self.new_cache()
            self.n_injected += 1
        return self.lm(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.lm, name)


@contextlib.contextmanager
def keep_kv(P):
    """P.lm -> KeepKV(P.lm) for the duration of the block; restored afterwards, also on an exception."""
    real = P.lm
    P.lm = KeepKV(real)
    try:
        yield P.lm
    finally:
        P.lm = real


# ------------------------------------------------------------------------------------------------ timing helpers
def warmup_set(reqs, images, pinpoints, n):
    """The first n requests plus the first request of every AnyRes grid not yet covered (crop counts 2, 3, 4)."""
    from transformers.image_processing_utils import select_best_resolution
    grid = lambda q: tuple(select_best_resolution((images[q["image_id"]]["height"], images[q["image_id"]]["width"]), pinpoints))
    out, seen = list(reqs[:n]), {grid(q) for q in reqs[:n]}
    for q in reqs[n:]:
        if grid(q) not in seen:
            out.append(q)
            seen.add(grid(q))
    return out


def _stats(v):
    v = np.asarray(v, float)
    if not len(v):
        return None
    return dict(mean=float(v.mean()), p50=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max()), n=int(len(v)))


# ------------------------------------------------------------------------------------------------ GPU exclusivity
def _num(x):
    try:
        return float(x)
    except ValueError:
        return None


def gpu_state():
    """Read-only nvidia-smi view: every GPU (index, uuid, utilisation %, memory MiB, name) and every compute process."""
    def q(what):
        out = subprocess.run(["nvidia-smi", f"--query-{what}", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=60, check=True).stdout
        return [l for l in out.splitlines() if l.strip()]
    gpus = []
    for l in q("gpu=index,uuid,utilization.gpu,memory.used,name"):
        idx, uuid, util, mem, name = [x.strip() for x in l.split(",", 4)]
        gpus.append(dict(index=int(idx), uuid=uuid, util=_num(util), mem_mib=_num(mem), name=name))
    procs = []
    for l in q("compute-apps=gpu_uuid,pid,used_memory,process_name"):
        uuid, pid, mem, name = [x.strip() for x in l.split(",", 3)]
        procs.append(dict(gpu=next((g["index"] for g in gpus if g["uuid"] == uuid), None), pid=int(pid), mem_mib=_num(mem), name=name))
    return gpus, procs


def exclusivity(own, during_run=False):
    """ok = no compute process other than ours on any GPU and no utilisation on any GPU other than ours (before our
    CUDA context exists, ours must be idle too)."""
    rec = dict(time=datetime.now().isoformat(timespec="seconds"), loadavg=list(os.getloadavg()))
    try:
        gpus, procs = gpu_state()
    except Exception as e:  # no nvidia-smi or a driver error: cannot show the GPUs are free
        return dict(rec, ok=False, error=repr(e))
    others = [p for p in procs if p["pid"] != os.getpid()]
    busy = [g["index"] for g in gpus if (g["util"] or 0) >= UTIL_BUSY and not (during_run and g["index"] == own)]
    return dict(rec, ok=not others and not busy, other_procs=others, busy_gpus=busy,
                gpus=[dict(index=g["index"], util=g["util"], mem_mib=g["mem_mib"]) for g in gpus],
                own_uuid=next((g["uuid"] for g in gpus if g["index"] == own), None))


def own_gpu():
    """Physical index of the one GPU this run may use (CUDA_VISIBLE_DEVICES, PCI bus order)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd.isdigit():
        raise SystemExit("Set CUDA_VISIBLE_DEVICES to one GPU index (e.g. CUDA_VISIBLE_DEVICES=0).")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # indices as nvidia-smi numbers them; CUDA is not initialised yet
    return int(cvd)
