"""VisionZip (640) on every round-0 CAL question, so that the 1,000 CAL/TEST rounds can count its excesses on any TEST
split. GPU; resumes. The reduced path is exactly exp_f4_reducers / exp_pope_reducers (same code, same packing); only the
question set (round-0 CAL instead of round-0 TEST) and the output folder change. Together with F4 / P4 (round-0 TEST)
this covers every non-DEV question, which is the pool every round draws its TEST split from.

    python -m exp.exp_reducers_cal --dataset sugarcrepe        # or textvqa, pope

Outputs: results/F7_reducers_cal/<dataset>_visionzip/traces.jsonl (+ run_config.json). The answers are deterministic;
the stage times are recorded but not used, because these runs may share the GPU with other jobs (the saving of
VisionZip is taken from the exclusive round-0 runs of F4 / P4)."""
import argparse
import json

from root import MANIFEST_DIR, RESULTS_DIR
from utils.tools import read_jsonl

OUT = RESULTS_DIR / "F7_reducers_cal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sugarcrepe", "textvqa", "pope"], required=True)
    ap.add_argument("--reducer", default="visionzip", choices=["visionzip", "pool"])
    a = ap.parse_args()
    if a.dataset == "pope":
        from exp import exp_pope_reducers as PR
        from exp import pope_common as C

        def round0_cal():
            cal = set(json.loads((C.LOCK_DIR / "splits_pope.json").read_text())["rounds"][0]["CAL"])
            return sorted((q for q in C.questions() if q["image_id"] in cal), key=lambda q: q["qid"])
        PR.round0_test, PR.OUT = round0_cal, OUT
        PR.run(a.reducer)
    else:
        from exp import exp_f4_reducers as F4

        def round0_cal(ds):
            cal = set(json.loads((F4.LOCK_DIR / f"splits_{ds}.json").read_text())["rounds"][0]["CAL"])
            return sorted((q for q in read_jsonl(MANIFEST_DIR / f"{ds}_questions.jsonl") if q["image_id"] in cal), key=lambda q: q["qid"])
        F4.round0_test_questions, F4.OUT = round0_cal, OUT
        F4.run(a.dataset, a.reducer)


if __name__ == "__main__":
    main()
