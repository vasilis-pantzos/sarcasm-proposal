"""Phase 3c: interventions that test the mechanism found in 3a/3b.

* pattern_value_patching: for heads of one layer, at the LAST token, recombine the attention pattern and the values
  of the clean and the corrupted run: is the effect of a head due to WHERE it looks (pattern) or to WHAT it reads
  (values)?
* mean_last / ablate_d: mean-ablate heads and/or the MLP of one layer at the last token (knockout).
* route_patching: patch the attention outputs of many layers at chosen positions (denoising: clean into corrupted;
  noising: corrupted into clean) — attention is the only way information moves between positions.
"""
import numpy as np
import torch

from .direction import _by_length
from .eap import _layers
from .flow import _last_row_attention
from .hooks import hooks, replace_positions, run_with_cache
from .metric import d_from_hidden, d_of, final_hidden, yn_ids
from .model import model_info


def _info(model):
    i = model_info(model)
    return i["n_heads"], i["n_kv_heads"], i["head_dim"]


@torch.no_grad()
def _pattern_and_values(model, ids, layer):
    """Attention pattern of the last token [H, T] and the values [T, KVH, hd] at `layer` for one run."""
    H, KVH, hd = _info(model)
    out = {}

    def pre(m, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        cos, sin = kwargs["position_embeddings"]
        out["p"] = _last_row_attention(m, h, cos, sin, H, KVH, hd)[0]
        out["v"] = m.v_proj(h)[0].view(h.shape[1], KVH, hd).float()
    hk = _layers(model)[layer].self_attn.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        final_hidden(model, torch.tensor([ids], device=model.device))
    finally:
        hk.remove()
    return out["p"], out["v"]


@torch.no_grad()
def pattern_value_patching(model, tok, pair, layer, head_sets):
    """Denoising at the last token of `layer`. For every named set of heads, three variants:
        full    = clean pattern x clean values   (the clean head output)
        pattern = clean pattern x corrupted values
        value   = corrupted pattern x clean values
    Returns {set_name: {variant: d}} (d of the corrupted input with those heads replaced)."""
    H, KVH, hd = _info(model)
    rep = H // KVH
    pc, vc = _pattern_and_values(model, pair.clean.ids, layer)
    rows = [(name, var) for name in head_sets for var in ("full", "pattern", "value")]
    state = {}

    def pre_attn(m, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        cos, sin = kwargs["position_embeddings"]
        state["p"] = _last_row_attention(m, h, cos, sin, H, KVH, hd)[0]           # same for all rows
        state["v"] = m.v_proj(h)[0].view(h.shape[1], KVH, hd).float()

    def pre_o(m, args, kwargs):
        z = (args[0] if args else kwargs["input"]).clone()
        pr, vr = state["p"], state["v"]
        for b, (name, var) in enumerate(rows):
            P = pc if var in ("full", "pattern") else pr
            V = vc if var in ("full", "value") else vr
            for hh in head_sets[name]:
                g = hh // rep
                z[b, -1, hh * hd:(hh + 1) * hd] = (P[hh] @ V[:, g]).to(z.dtype)
        if args:
            return (z,) + tuple(args[1:]), kwargs
        kwargs["input"] = z
        return args, kwargs

    att = _layers(model)[layer].self_attn
    h1 = att.register_forward_pre_hook(pre_attn, with_kwargs=True)
    h2 = att.o_proj.register_forward_pre_hook(pre_o, with_kwargs=True)
    try:
        x = torch.tensor([pair.corrupted.ids] * len(rows), device=model.device)
        d = d_of(model, tok, x).numpy()
    finally:
        h1.remove(); h2.remove()
    out = {}
    for (name, var), v in zip(rows, d):
        out.setdefault(name, {})[var] = float(v)
    return out


@torch.no_grad()
def mean_last(model, id_lists, layer, batch=32):
    """Mean over prompts of the per-head outputs z [H*hd] and of the MLP output [d] at the last token of `layer`."""
    store, zs, ms = {}, [], []
    lay = _layers(model)[layer]
    h1 = lay.self_attn.o_proj.register_forward_pre_hook(lambda m, a: store.__setitem__("z", a[0][:, -1].float()))
    h2 = lay.mlp.register_forward_hook(lambda m, a, o: store.__setitem__("m", o[:, -1].float()))
    try:
        for idx in _by_length(id_lists, batch):
            final_hidden(model, torch.tensor([id_lists[i] for i in idx], device=model.device))
            zs.append(store["z"].sum(0)); ms.append(store["m"].sum(0))
    finally:
        h1.remove(); h2.remove()
    n = len(id_lists)
    return (torch.stack(zs).sum(0) / n), (torch.stack(ms).sum(0) / n)


@torch.no_grad()
def ablate_d(model, tok, id_lists, layer, heads=(), mlp=False, zmean=None, mmean=None, batch=32):
    """d of every prompt when the given heads (and/or the MLP) of `layer` are replaced by their mean at the last token."""
    H, KVH, hd = _info(model)
    Y, N = yn_ids(tok)
    lay = _layers(model)[layer]
    hs = []
    if len(heads):
        def pre(m, args):
            z = args[0].clone()
            for hh in heads:
                z[:, -1, hh * hd:(hh + 1) * hd] = zmean[hh * hd:(hh + 1) * hd].to(z.dtype)
            return (z,) + tuple(args[1:])
        hs.append(lay.self_attn.o_proj.register_forward_pre_hook(pre))
    if mlp:
        def post(m, a, o):
            o = o.clone()
            o[:, -1] = mmean.to(o.dtype)
            return o
        hs.append(lay.mlp.register_forward_hook(post))
    out = np.zeros(len(id_lists), dtype=np.float32)
    try:
        for idx in _by_length(id_lists, batch):
            x = torch.tensor([id_lists[i] for i in idx], device=model.device)
            out[idx] = d_from_hidden(model, final_hidden(model, x), Y, N).cpu().numpy()
    finally:
        for h in hs:
            h.remove()
    return out


@torch.no_grad()
def route_patching(model, tok, pair, routes, mode="denoise"):
    """routes: {name: [(layer, positions), ...]}. Patch the attention output at those layers and positions.
    mode "denoise": clean values into the corrupted run; "noise": corrupted values into the clean run.
    Returns {name: d}."""
    src, dst = (pair.clean, pair.corrupted) if mode == "denoise" else (pair.corrupted, pair.clean)
    L = len(_layers(model))
    _, cache = run_with_cache(model, src.ids, kinds=("attn",))
    out = {}
    for name, spec in routes.items():
        specs = [("attn", l, replace_positions(cache[("attn", l)], [pos])) for l, pos in spec if len(pos) and l < L]
        with hooks(model, specs):
            out[name] = float(d_of(model, tok, torch.tensor([dst.ids], device=model.device))[0])
    return out
