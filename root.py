from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()

# Inputs (data/) are kept apart from outputs (results/, logs/); code lives in exp/, models/, data_provider/, utils/.
DATA_DIR = ROOT_DIR / "data"
MANIFEST_DIR = DATA_DIR / "manifests"      # frozen question/image manifests and splits
RESULTS_DIR = ROOT_DIR / "results"
LOG_DIR = ROOT_DIR / "logs"

# The DEV paired runs. A run with a different configuration gets a new sibling folder with a
# suffix; existing folders are never overwritten.
E2_DIR = RESULTS_DIR / "E2_sugarcrepe_paired_dev"
E5_DIR = RESULTS_DIR / "E5_textvqa_paired_dev"


def results_dir(name, default):
    """results/<name> when a run names its own folder (a new configuration); otherwise the canonical folder."""
    return RESULTS_DIR / name if name else default

