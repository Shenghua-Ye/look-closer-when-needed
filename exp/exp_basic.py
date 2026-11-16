import os
import random
import sys
from datetime import datetime

import numpy as np

from root import LOG_DIR


class _Tee(object):
    """Write stdout to the terminal and to a log file."""

    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "a", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


class ExpBasic(object):
    def __init__(self, args):
        self.args = args
        self.setting = self._setting()
        self._setup_logger()
        self._acquire_device()
        self._set_seed(args.seed)

    def _setting(self):
        setting = "{}({})_{}_n{}".format(
            self.args.task_name,
            self.args.dataset,
            self.args.model,
            self.args.limit if self.args.limit else "all",
        )
        if getattr(self.args, "exp_dir", ""):
            setting += "_" + self.args.exp_dir
        if getattr(self.args, "loss", "strict") != "strict":
            setting += "_loss-" + self.args.loss
        if getattr(self.args, "signal", "first_margin") != "first_margin":
            setting += "_signal-" + self.args.signal
        return setting

    def _acquire_device(self):
        # Experiments address the GPU as cuda:0; --gpu picks the physical card unless the script already set it.
        if self.args.use_gpu:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.args.gpu))
            print(f"Use GPU: {os.environ['CUDA_VISIBLE_DEVICES']} (cuda:0 inside the process)")
        else:
            print("Use CPU")

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
        except ImportError:  # CPU-only analysis environments
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Data sampling uses its own fixed seeds (see data_provider/ and each exp module); this seed only fixes
        # any remaining library randomness.

    def _setup_logger(self):
        """Tee stdout and stderr (warnings, tracebacks) to logs/<task_name>/<setting>_<timestamp>.log"""
        self.log_path = None
        if not self.args.is_logging:
            return
        log_dir = LOG_DIR / self.args.task_name
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"{self.setting}_{timestamp}.log"
        sys.stdout = _Tee(sys.stdout, self.log_path)
        sys.stderr = _Tee(sys.stderr, self.log_path)
        print(f"[{datetime.now()}] Experiment setting: {self.setting}")
        print(f"Log file created: {self.log_path}")
        print("Args: " + ", ".join(f"{k}={v}" for k, v in sorted(vars(self.args).items())))

    def run(self):
        raise NotImplementedError
