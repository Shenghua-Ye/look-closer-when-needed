from root import MANIFEST_DIR

# Dataset registry: frozen manifests in data/manifests/ (built by data_provider/manifest_*.py).
# task: how answers are produced and compared.
#   ab_scoring   -- two-caption choice scored from next-token log-probabilities of "A" / "B" (no decoding)
#   short_answer -- greedy short-answer generation, compared after VQA answer normalisation
datasets = {
    "sugarcrepe": dict(
        questions=MANIFEST_DIR / "sugarcrepe_questions.jsonl",
        images=MANIFEST_DIR / "sugarcrepe_images.jsonl",
        meta=MANIFEST_DIR / "manifest_meta.json",
        task="ab_scoring",
        group_key="obs_group",          # observable caption-edit group (routing); "category" is diagnostic only
    ),
    "textvqa": dict(
        questions=MANIFEST_DIR / "textvqa_questions.jsonl",
        images=MANIFEST_DIR / "textvqa_images.jsonl",
        meta=MANIFEST_DIR / "textvqa_meta.json",
        task="short_answer",
        group_key="qtype",              # question-type rules v1 on the question text
    ),
}
