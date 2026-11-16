"""Cheap (base view) and full (native AnyRes) inference paths of LLaVA-NeXT-7B for SugarCrepe A/B option scoring.

full_native: the fixed reference. Processor + model forward exactly as shipped (expanded image tokens, AnyRes
             packing with unpadding and newline tokens). Nothing is reused from the cheap path (rho = 0).
base_*:      the cheap path. Only the base 336x336 view: 576 projected patch features plus one image_newline
             token, which is how the model itself packs an image that has a single patch. The language model is
             called with hand-built input embeddings (prompt prefix, image features, prompt suffix).

Answers: the log-probabilities of the "A" and "B" tokens at the next position are
aggregated over the two tokenizer variants; the higher one is the answer and |log pA - log pB| is the margin.
"""
import numpy as np
import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

MODEL = "llava-hf/llava-v1.6-vicuna-7b-hf"
REVISION = "c916e6cdcd760b4cecd1dd4907f84ac649f93b23"
QUESTION = ("Which caption better describes the image?\nA. {a}\nB. {b}\n"
            "Answer with the option's letter from the given choices directly.")
OPEN_SUFFIX = "\nAnswer the question using a single word or phrase."  # LLaVA short-answer prompt
MAX_NEW = 32
EOS_ID = 2  # </s>; the model's generation_config has no other logits processors, so greedy = argmax


class NextPaths:
    def __init__(self, device="cuda:0", attn="sdpa"):
        self.dev = device
        self.attn = attn
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL, revision=REVISION, torch_dtype=torch.float16, attn_implementation=attn).to(device).eval().requires_grad_(False)
        self.proc = LlavaNextProcessor.from_pretrained(MODEL, revision=REVISION)
        self.tok = self.proc.tokenizer
        self.emb = self.model.get_input_embeddings()
        self.lm = self.model.language_model
        self.img_tok = self.model.config.image_token_index
        ip = self.proc.image_processor
        self.mean = torch.tensor(ip.image_mean).view(3, 1, 1)
        self.std = torch.tensor(ip.image_std).view(3, 1, 1)
        self.base_size = ip.crop_size["height"] if isinstance(ip.crop_size, dict) else 336
        unk = self.tok.unk_token_id
        self.a_ids = [self.tok.convert_tokens_to_ids(t) for t in ("▁A", "A")]
        self.b_ids = [self.tok.convert_tokens_to_ids(t) for t in ("▁B", "B")]
        assert unk not in self.a_ids + self.b_ids, (self.a_ids, self.b_ids)

    # ------------------------------------------------------------------ prompt
    def prompt(self, option_a, option_b):
        conv = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": QUESTION.format(a=option_a, b=option_b)}]}]
        return self.proc.apply_chat_template(conv, add_generation_prompt=True)

    def _split_ids(self, prompt):
        ids = self.tok(prompt, return_tensors="pt").input_ids[0].to(self.dev)
        pos = (ids == self.img_tok).nonzero()
        assert len(pos) == 1, "prompt must contain exactly one <image> placeholder before expansion"
        cut = int(pos[0])
        return ids[:cut], ids[cut + 1:]

    def score(self, logits):
        lp = torch.log_softmax(logits.float(), -1)
        la = torch.logsumexp(lp[self.a_ids], -1)
        lb = torch.logsumexp(lp[self.b_ids], -1)
        return ("A" if la > lb else "B"), float((la - lb).abs()), float(la), float(lb)

    # ------------------------------------------------------------------ full path (native reference)
    @torch.no_grad()
    def full_inputs(self, image, prompt):
        inputs = self.proc(images=image, text=prompt, return_tensors="pt").to(self.dev)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        return inputs

    @torch.no_grad()
    def full_native(self, image, prompt):
        inputs = self.full_inputs(image, prompt)
        logits = self.model(**inputs, use_cache=False).logits[0, -1]
        n_img = int((inputs["input_ids"] == self.img_tok).sum())
        return logits, n_img, int(inputs["pixel_values"].shape[1])

    @torch.no_grad()
    def full_manual(self, image, prompt):
        """Native features through our hand-built embeddings: must reproduce full_native (checks the machinery)."""
        inputs = self.full_inputs(image, prompt)
        cfg = self.model.config
        feats = self.model.get_image_features(inputs["pixel_values"], inputs["image_sizes"],
                                              vision_feature_layer=cfg.vision_feature_layer,
                                              vision_feature_select_strategy=cfg.vision_feature_select_strategy)
        packed, _ = self.model.pack_image_features(feats, inputs["image_sizes"], cfg.vision_feature_select_strategy,
                                                   image_newline=self.model.image_newline)
        return self.prefill(packed, prompt).logits[0, -1], int(packed.shape[0])

    # ------------------------------------------------------------------ cheap path (base view only)
    def base_pixel(self, image):
        """Base view preprocessing: squash to 336x336 (bicubic), rescale, normalize -- as the processor's first patch."""
        im = image.convert("RGB").resize((self.base_size, self.base_size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return ((x - self.mean) / self.std)[None].to(self.dev, torch.float16)

    @torch.no_grad()
    def base_encode(self, pix):
        """Vision tower + projector on the base view. Returns packed features (577 x d) and the mean tower feature."""
        cfg = self.model.config
        hs = self.model.vision_tower(pix, output_hidden_states=True).hidden_states[cfg.vision_feature_layer]
        if cfg.vision_feature_select_strategy == "default":
            hs = hs[:, 1:]
        feats = self.model.multi_modal_projector(hs)[0]
        packed = torch.cat([feats, self.model.image_newline[None].to(feats)], 0)
        return packed, hs[0].float().mean(0)

    @torch.no_grad()
    def prefill(self, packed, prompt, use_cache=False):
        pre, post = self._split_ids(prompt)
        x = torch.cat([self.emb(pre), packed.to(self.emb.weight.dtype), self.emb(post)])[None]
        return self.lm(inputs_embeds=x, use_cache=use_cache)

    # ------------------------------------------------------------------ short-answer generation (TextVQA)
    def prompt_open(self, question):
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question + OPEN_SUFFIX}]}]
        return self.proc.apply_chat_template(conv, add_generation_prompt=True)

    @staticmethod
    def first_margin(logits):
        """Top-1 minus top-2 softmax probability at the first generation position, and the top-1 token."""
        top = torch.topk(torch.softmax(logits.float(), -1), 2)
        return float(top.values[0] - top.values[1]), int(top.indices[0])

    def decode_text(self, tokens):
        return self.tok.decode(tokens, skip_special_tokens=True).strip()

    @torch.no_grad()
    def full_generate(self, image, prompt, max_new=None):
        """Native AnyRes greedy generation: the fixed reference answer."""
        max_new = max_new or MAX_NEW
        inputs = self.full_inputs(image, prompt)
        n_in = inputs["input_ids"].shape[1]
        out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False, output_logits=True,
                                  return_dict_in_generate=True, pad_token_id=self.tok.pad_token_id)
        new = out.sequences[0, n_in:].tolist()
        m, top = self.first_margin(out.logits[0][0])
        n_img = int((inputs["input_ids"] == self.img_tok).sum())
        return dict(text=self.decode_text(new), tokens=new, margin=m, n_new=len(new),
                    truncated=len(new) >= max_new and new[-1] != EOS_ID, n_visual_tokens=n_img)

    @torch.no_grad()
    def greedy_from_embeddings(self, packed, prompt, max_new=None, lam=None, clock=None):
        """Prefill with the given visual features, read the first-position margin, then (unless m <= lam, the
        probe's fallback) continue greedy decoding from the same KV cache. clock(name) marks stage ends."""
        max_new = max_new or MAX_NEW
        out = self.prefill(packed, prompt, use_cache=True)
        m, nxt = self.first_margin(out.logits[0, -1])
        if clock:
            clock("prefill")
        if lam is not None and m <= lam:
            return dict(text=None, tokens=[], margin=m, n_new=0, truncated=False, fallback=True)
        toks, past = [nxt], out.past_key_values
        while len(toks) < max_new and toks[-1] != EOS_ID:
            o = self.lm(inputs_embeds=self.emb(torch.tensor([[toks[-1]]], device=self.dev)), past_key_values=past,
                        use_cache=True)
            past = o.past_key_values
            toks.append(int(o.logits[0, -1].argmax()))
        if clock:
            clock("decode")
        return dict(text=self.decode_text(toks), tokens=toks, margin=m, n_new=len(toks),
                    truncated=len(toks) >= max_new and toks[-1] != EOS_ID, fallback=False)

    @torch.no_grad()
    def full_packed(self, image, prompt):
        """Native AnyRes packed features (for checking the manual decode loop against native generate)."""
        inputs = self.full_inputs(image, prompt)
        cfg = self.model.config
        feats = self.model.get_image_features(inputs["pixel_values"], inputs["image_sizes"],
                                              vision_feature_layer=cfg.vision_feature_layer,
                                              vision_feature_select_strategy=cfg.vision_feature_select_strategy)
        packed, _ = self.model.pack_image_features(feats, inputs["image_sizes"], cfg.vision_feature_select_strategy,
                                                   image_newline=self.model.image_newline)
        return packed
