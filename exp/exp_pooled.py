"""Fixed-budget token reducers used as baselines (exp_f4_reducers, exp_pope_reducers, exp_reducers_cal).

  pool       every AnyRes crop is kept, but each crop's 24x24 grid of projected features is average-pooled to 12x12
             (2x2 pooling) and the crops are packed as LLaVA-NeXT does (base crop first; the high-resolution grid
             unpadded, one newline per row). Before a run, check_packing confirms that this packing code with pooling 1
             reproduces LLaVA-NeXT's own packing exactly.
  visionzip  VisionZip (Yang et al., CVPR 2025), ported from the official code (github.com/dvlab-research/VisionZip,
             Apache-2.0: visionzip/clip_encoder.py and llava_arch.py) to transformers' LLaVA-NeXT. Per crop, the
             penultimate CLIP layer keeps the CLS token and the 107 tokens with the most CLS attention (dominant) and
             merges the rest into 20 contextual tokens by key similarity: 128 tokens per crop, 640 for a 5-crop image
             (VisionZip's 640-token LLaVA-NeXT setting). Packing follows VisionZip's LLaVA-NeXT code: base crop, then
             the high-resolution dominant tokens in grid order (no unpadding, no newlines), each crop's CLS token, and
             the contextual tokens. The vision tower must return attention weights, so it runs eager attention, as in
             the official code.
"""
import torch
import torch.nn.functional as F

POOL = 2
VZ_DOMINANT, VZ_CONTEXTUAL = 108, 20  # per crop, CLS included in the dominant tokens


def pooled_pack(P, pixel_values, image_sizes, pool=POOL):
    """Projected features of all crops, pooled `pool` x `pool` per crop, packed like LlavaNext.pack_image_features.
    Returns the packed features and the base crop's mean tower feature (the Gate-1 feature of E2/E5)."""
    from transformers.models.llava_next.modeling_llava_next import get_anyres_image_grid_shape, unpad_image
    cfg = P.model.config
    hs = P.model.vision_tower(pixel_values[0], output_hidden_states=True).hidden_states[cfg.vision_feature_layer]
    if cfg.vision_feature_select_strategy == "default":
        hs = hs[:, 1:]
    proj = P.model.multi_modal_projector(hs)
    n, t, d = proj.shape
    side = int(round(t ** 0.5))
    s2 = side // pool
    g = proj.view(n, side, side, d).permute(0, 3, 1, 2)
    if pool > 1:
        g = F.avg_pool2d(g.float(), pool).to(proj.dtype)
    base = g[0].flatten(1).transpose(0, 1)
    nl = P.model.image_newline
    if n == 1:
        packed = torch.cat([base, nl[None].to(base)], 0)
    else:
        nph, npw = get_anyres_image_grid_shape(image_sizes[0], cfg.image_grid_pinpoints, cfg.vision_config.image_size)
        rest = g[1:].permute(0, 2, 3, 1).reshape(nph, npw, s2, s2, d).permute(4, 0, 2, 1, 3).reshape(d, nph * s2, npw * s2)
        rest = unpad_image(rest, image_sizes[0])
        rest = torch.cat([rest, nl[:, None, None].expand(d, rest.shape[1], 1).to(rest.dtype)], -1)
        packed = torch.cat([base, rest.flatten(1, 2).transpose(0, 1)], 0)
    return packed, hs[0].float().mean(0)


def visionzip_select(hidden_states, attn_weights, metric, dominant_num, contextual_num):
    """VisionZip's token selection, ported line by line from CLIPVisionTower_VisionZip.forward (batched over crops).
    dominant_num excludes the CLS token, which is always kept. Returns the kept tokens (dominant in index order, then
    contextual) and the kept indices (CLS first)."""
    B, T, D = hidden_states.shape
    cls_attention_sum = attn_weights[:, :, 0, 1:].sum(dim=1)
    topk_indices = cls_attention_sum.topk(dominant_num, dim=1).indices + 1
    all_indices = torch.cat([torch.zeros((B, 1), dtype=topk_indices.dtype, device=topk_indices.device), topk_indices], dim=1)
    mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool).scatter_(1, all_indices, False)
    dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(B, dominant_num + 1, D)
    metric_filtered = metric[mask].view(B, T - (dominant_num + 1), metric.shape[2])
    hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(B, T - (dominant_num + 1), D)
    metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True)
    step = max(1, metric_normalized.shape[1] // contextual_num)
    target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
    target_tokens = metric_normalized[:, target_indices, :]
    others = ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices)
    tokens_to_merge = metric_normalized[:, others, :]
    similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
    assign_one_hot = torch.zeros(B, tokens_to_merge.shape[1], contextual_num, dtype=hidden_states_filtered.dtype, device=metric_normalized.device)
    assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
    aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_states_filtered[:, others, :]) / counts
    contextual_tokens = hidden_states_filtered[:, target_indices, :] + aggregated_hidden
    return torch.cat([dominant_tokens, contextual_tokens], dim=1), all_indices


def visionzip_restore(image_feature, keep_idx, nph, npw, side=24):
    """VisionZip's restore_image_features_sorted for the high-resolution crops: dominant tokens in grid order, then
    each crop's CLS token, then the contextual tokens."""
    num_img, total, d = image_feature.shape
    num_keep = keep_idx.shape[1]
    num_extra = total - num_keep
    keep_sorted, _ = keep_idx.sort(dim=1)
    pos = keep_sorted[:, 1:] - 1
    restored = torch.zeros((num_img, side * side, d), device=image_feature.device, dtype=image_feature.dtype)
    mask = torch.zeros(num_img, side * side, dtype=torch.bool, device=image_feature.device)
    mask.scatter_(1, pos, True)
    restored[mask] = image_feature[:, 1:num_keep, :].reshape(-1, d)
    restored = restored.view(nph, npw, side, side, d).permute(0, 2, 1, 3, 4).reshape(nph * side, npw * side, d)
    mask = mask.view(nph, npw, side, side).permute(0, 2, 1, 3).reshape(nph * side, npw * side)
    return torch.cat([restored[mask], image_feature[:, 0], image_feature[:, -num_extra:].reshape(-1, d)])


def visionzip_pack(P, pixel_values, image_sizes, dominant=VZ_DOMINANT, contextual=VZ_CONTEXTUAL):
    """VisionZip features of all crops, packed as VisionZip's LLaVA-NeXT code does. Also returns the base crop's mean
    patch feature (the Gate-1 feature of E2/E5)."""
    from transformers.models.llava_next.modeling_llava_next import get_anyres_image_grid_shape
    cfg = P.model.config
    vt = P.model.vision_tower
    attn_layer = vt.vision_model.encoder.layers[cfg.vision_feature_layer].self_attn
    cap = {}
    def hook(mod, args, kwargs):  # VisionZip's metric: the layer's key states, averaged over heads
        h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        b, t, _ = h.shape
        cap["metric"] = mod.k_proj(h).view(b, t, mod.num_heads, mod.head_dim).mean(2)
    handle = attn_layer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        out = vt(pixel_values[0], output_hidden_states=True, output_attentions=True)
    finally:
        handle.remove()
    hs, attn = out.hidden_states[cfg.vision_feature_layer], out.attentions[cfg.vision_feature_layer]
    sel, keep = visionzip_select(hs, attn, cap["metric"], dominant - 1, contextual)
    feats = P.model.multi_modal_projector(sel)
    if feats.shape[0] == 1:
        packed = torch.cat([feats[0], P.model.image_newline[None].to(feats)], 0)
    else:
        nph, npw = get_anyres_image_grid_shape(image_sizes[0], cfg.image_grid_pinpoints, cfg.vision_config.image_size)
        packed = torch.cat([feats[0], visionzip_restore(feats[1:], keep[1:], nph, npw)], 0)
    return packed, hs[0, 1:].float().mean(0)


REDUCERS = {"pool": pooled_pack, "visionzip": visionzip_pack}
CHEAP_PATHS = {"pool": f"pooled AnyRes: {POOL}x{POOL} average pooling of the projected features of every crop, LLaVA-NeXT packing",
               "visionzip": f"VisionZip on every crop: {VZ_DOMINANT} dominant (CLS included) + {VZ_CONTEXTUAL} contextual tokens, "
                            "VisionZip's LLaVA-NeXT packing; ported from the official code"}


def image_inputs(P, img):
    ip = P.proc.image_processor(img, return_tensors="pt")
    return ip["pixel_values"].to(P.dev, torch.float16), ip["image_sizes"]


@torch.inference_mode()
def check_packing(P, items):
    """With pooling 1, our packing must equal LLaVA-NeXT's own (max abs difference 0) on these items."""
    worst = 0.0
    for img, prompt in items:
        pv, sizes = image_inputs(P, img)
        ours, _ = pooled_pack(P, pv, sizes, pool=1)
        ref = P.full_packed(img, prompt)
        if ours.shape != ref.shape:
            raise SystemExit(f"packing check failed: shape {tuple(ours.shape)} vs {tuple(ref.shape)}")
        worst = max(worst, float((ours.float() - ref.float()).abs().max()))
    print(f"packing check (pool 1 vs LLaVA-NeXT): {len(items)} images, max abs diff {worst}", flush=True)
    if worst != 0.0:
        raise SystemExit("packing check failed: pooled packing does not reproduce the model's packing")
