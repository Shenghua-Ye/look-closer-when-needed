import argparse
import json
from pathlib import Path


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def check_run_config(path, cfg):
    """Write run_config.json on the first run. On resume, refuse to continue under a different configuration, so
    one results folder never mixes configurations (new configuration -> new folder via --exp_dir)."""
    cfg = json.loads(json.dumps(cfg))
    if not Path(path).exists():
        Path(path).write_text(json.dumps(cfg, indent=1))
        return
    old = json.loads(Path(path).read_text())
    diff = sorted(k for k in set(old) | set(cfg) if old.get(k) != cfg.get(k))
    if diff:
        raise SystemExit(f"{path} records a different configuration (differs in: {', '.join(diff)}). "
                         "Run the new configuration in its own results folder: --exp_dir <name>")
    print(f"run_config.json matches: resuming {Path(path).parent}")


def sync():
    """Synchronise CUDA before reading a timer, so stage times are real GPU times."""
    import torch
    torch.cuda.synchronize()
