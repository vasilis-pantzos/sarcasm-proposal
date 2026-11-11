"""Phase 3b: where the information goes. Attention of the LAST token, per head and per region of the prompt,
and what each head writes along a direction, split by the region it reads from.

The attention weights are recomputed from the queries and keys (with the rotary position embedding), so this
does not depend on the attention implementation of the installed transformers version. Only the last query row
is computed: p[b, h, j] = softmax_j(q_last . k_j / sqrt(head_dim)).
"""
import numpy as np
import torch

from .eap import _layers
from .direction import _o_weight32
from .metric import final_hidden
from .model import model_info
from .prompts import REGIONS


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _last_row_attention(attn, h, cos, sin, H, KVH, hd):
    """Attention probabilities of the last query over all positions: [B, H, T]. h: input of self_attn [B, T, d]."""
    B, T, _ = h.shape
    q = attn.q_proj(h[:, -1:]).view(B, 1, H, hd).transpose(1, 2).float()          # [B, H, 1, hd]
    k = attn.k_proj(h).view(B, T, KVH, hd).transpose(1, 2).float()                # [B, KVH, T, hd]
    c, s = cos.float().unsqueeze(1), sin.float().unsqueeze(1)                       # [B, 1, T, hd]
    q = q * c[:, :, -1:] + _rotate_half(q) * s[:, :, -1:]
    k = k * c + _rotate_half(k) * s
    k = k.repeat_interleave(H // KVH, dim=1)                                        # [B, H, T, hd]
    scores = (q @ k.transpose(-1, -2)).squeeze(2) / hd ** 0.5                       # [B, H, T]
    return scores.softmax(-1)


def region_masks(encs, T):
    """[n, T, len(REGIONS) + 1] one-hot of the region of every position (last column = everything else)."""
    M = np.zeros((len(encs), T, len(REGIONS) + 1), dtype=np.float32)
    M[:, :, -1] = 1
    for i, e in enumerate(encs):
        for r, name in enumerate(REGIONS):
            for p in e.regions[name]:
                M[i, p, r] = 1
                M[i, p, -1] = 0
    return M


@torch.no_grad()
def attention_by_region(model, encs, direction=None, write_layers=(), batch=32, progress=True):
    """For every prompt (Encoded), every layer and head: the attention of the last token to each region
    (situation, phrase, question, final, other) -> attn [n, L, H, 5].

    If `direction` is given: for the layers in `write_layers`, what each head writes along the (unit) direction at
    the last token, split by the region it reads from -> writes [n, len(write_layers), H, 5]
    (sum over regions = the head's projection on the direction)."""
    info = model_info(model)
    L, H, hd = info["n_layers"], info["n_heads"], info["head_dim"]
    KVH = info["n_kv_heads"]
    R = len(REGIONS) + 1
    n = len(encs)
    A = np.zeros((n, L, H, R), dtype=np.float32)
    W = np.zeros((n, len(write_layers), H, R), dtype=np.float32)
    if direction is not None:
        u = torch.as_tensor(np.asarray(direction) / np.linalg.norm(direction), dtype=torch.float32, device=model.device)
        Wu = {l: (_o_weight32(model, l).view(-1, H, hd) * u[:, None, None]).sum(0) for l in write_layers}   # [H, hd]
    groups = {}
    for i, e in enumerate(encs):
        groups.setdefault(len(e.ids), []).append(i)
    store = {}
    hooks = []
    for l, layer in enumerate(_layers(model)):
        def pre(m, args, kwargs, l=l):
            h = args[0] if args else kwargs["hidden_states"]
            cos, sin = kwargs["position_embeddings"]
            p = _last_row_attention(m, h, cos, sin, H, KVH, hd)
            store[("p", l)] = p
            if l in write_layers:
                B, T, _ = h.shape
                v = m.v_proj(h).view(B, T, KVH, hd).float().repeat_interleave(H // KVH, dim=2)   # [B, T, H, hd]
                store[("vu", l)] = (v * Wu[l][None, None]).sum(-1).transpose(1, 2)                 # [B, H, T]
        hooks.append(layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        done = 0
        for T, idx in sorted(groups.items()):
            for j in range(0, len(idx), batch):
                part = idx[j:j + batch]
                x = torch.tensor([encs[i].ids for i in part], device=model.device)
                Mk = torch.as_tensor(region_masks([encs[i] for i in part], T), device=model.device)   # [B, T, R]
                final_hidden(model, x)
                for l in range(L):
                    A[part, l] = torch.einsum("bht,btr->bhr", store[("p", l)], Mk).cpu().numpy()
                for k, l in enumerate(write_layers):
                    W[part, k] = torch.einsum("bht,btr->bhr", store[("p", l)] * store[("vu", l)], Mk).cpu().numpy()
                done += len(part)
                if progress and done % 320 < len(part):
                    print(f"  {done}/{n} prompts")
    finally:
        for hk in hooks:
            hk.remove()
    return (A, W) if direction is not None else A


@torch.no_grad()
def attention_row(model, ids, layer, head):
    """Attention of the last token over all positions for one prompt, one head: [T]."""
    info = model_info(model)
    H, hd, KVH = info["n_heads"], info["head_dim"], info["n_kv_heads"]
    out = {}

    def pre(m, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        cos, sin = kwargs["position_embeddings"]
        out["p"] = _last_row_attention(m, h, cos, sin, H, KVH, hd)[0, head]
    hk = _layers(model)[layer].self_attn.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        final_hidden(model, torch.tensor([ids], device=model.device))
    finally:
        hk.remove()
    return out["p"].cpu().numpy()
