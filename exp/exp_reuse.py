"""Resume a fallback from the cheap pass instead of restarting it (the Reuse class), used by the paired runs and the
timing runs (exp_f1_paired, exp_f3_timing, exp_pope_paired, exp_pope_timing).

LLaVA-NeXT packs the 576 base-view tokens before the AnyRes crop tokens. The cheap input
[prefix][base 576][newline][question] and the full input [prefix][base 576][crop tokens][question] therefore share
their first len(prefix) + 576 tokens. On a fallback the cheap pass's KV cache is cropped to that shared prefix and only
the crop tokens and the question are prefilled. The vision tower runs on the crops only; the base slot reuses the cheap
features. On TextVQA the cheap answer is also a draft: one forward pass checks every draft token against the full
model's greedy choice, so the returned answer is still the full path's greedy answer.
"""
import time

import torch
from transformers import DynamicCache
from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches

from utils.tools import sync


class Reuse:
    """Cheap pass with a returnable KV cache, and the continuation that turns it into the full path."""

    def __init__(self, P):
        self.P = P
        self.cfg = P.model.config

    @torch.no_grad()
    def cheap(self, img, prompt):
        P = self.P
        pre, post = P._split_ids(prompt)
        packed, _ = P.base_encode(P.base_pixel(img))
        x = torch.cat([P.emb(pre), packed.to(P.emb.weight.dtype), P.emb(post)])[None]
        out = P.lm(inputs_embeds=x, past_key_values=DynamicCache(), use_cache=True)
        return out, packed, len(pre), post

    @torch.no_grad()
    def greedy(self, first_logits, cache, max_new):
        P = self.P
        from models.llava_next import EOS_ID
        toks = [int(torch.topk(torch.softmax(first_logits.float(), -1), 1).indices[0])]
        while len(toks) < max_new and toks[-1] != EOS_ID:
            o = P.lm(inputs_embeds=P.emb(torch.tensor([[toks[-1]]], device=P.dev)), past_key_values=cache,
                     use_cache=True)
            cache = o.past_key_values
            toks.append(int(o.logits[0, -1].argmax()))
        return toks

    @torch.no_grad()
    def crop_features(self, img, prompt, base_packed):
        """Processor preprocessing, vision tower on the crops only, packing with the cheap base features."""
        P, cfg = self.P, self.cfg
        inputs = P.full_inputs(img, prompt)
        sync()
        t_pre = time.perf_counter()
        pix = inputs["pixel_values"][0]
        n = int(image_size_to_num_patches(image_size=inputs["image_sizes"][0], grid_pinpoints=cfg.image_grid_pinpoints,
                                          patch_size=cfg.vision_config.image_size))
        hs = P.model.vision_tower(pix[1:n], output_hidden_states=True).hidden_states[cfg.vision_feature_layer]
        if cfg.vision_feature_select_strategy == "default":
            hs = hs[:, 1:]
        crops = P.model.multi_modal_projector(hs)
        feats = torch.cat([base_packed[:576][None].to(crops.dtype), crops])
        packed, _ = P.model.pack_image_features([feats], inputs["image_sizes"], cfg.vision_feature_select_strategy,
                                                image_newline=P.model.image_newline)
        return packed, t_pre

    @torch.no_grad()
    def continue_prefill(self, cache, n_keep, packed_full, post):
        P = self.P
        cache.crop(n_keep)
        tail = torch.cat([packed_full[576:].to(P.emb.weight.dtype), P.emb(post)])[None]
        return P.lm(inputs_embeds=tail, past_key_values=cache, use_cache=True)

    @torch.no_grad()
    def verify_draft(self, first_logits, cache, draft, max_new):
        """The full model's greedy answer, checking the cheap draft in one forward pass.
        Returns (tokens, number of draft tokens accepted)."""
        P = self.P
        from models.llava_next import EOS_ID
        n0 = cache.get_seq_length()
        pred0 = int(torch.topk(torch.softmax(first_logits.float(), -1), 1).indices[0])
        if not draft or pred0 != draft[0]:
            return self.greedy(first_logits, cache, max_new), 0
        k = len(draft)
        preds = [pred0]
        if k > 1:
            o = P.lm(inputs_embeds=P.emb(torch.tensor([draft[:-1]], device=P.dev)), past_key_values=cache,
                     use_cache=True)
            preds += [int(x) for x in o.logits[0].argmax(-1).tolist()]
            cache = o.past_key_values
        j = next((i for i in range(k) if preds[i] != draft[i]), k)
        if j == k and (draft[-1] == EOS_ID or k >= max_new):
            return list(draft), k
        cache.crop(n0 + j)
        toks = list(draft[:j]) + ([preds[j]] if j < k else [])
        while len(toks) < max_new and toks[-1] != EOS_ID:
            o = P.lm(inputs_embeds=P.emb(torch.tensor([[toks[-1]]], device=P.dev)), past_key_values=cache,
                     use_cache=True)
            cache = o.past_key_values
            toks.append(int(o.logits[0, -1].argmax()))
        return toks, j
