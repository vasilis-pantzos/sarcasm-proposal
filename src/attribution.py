"""Attribution patching for nodes (attention heads and MLP layers).

A linear (first-order) estimate of activation patching that needs only one forward pass on the
clean input and one forward + backward pass on the corrupted input, for ALL nodes at once:

    estimated effect of patching node n at positions P
        = sum_{p in P} (a_clean[p] - a_corr[p]) . d(d)/d(a)[p]   (gradient taken on the corrupted run)

Divided by d(S) - d(corrupted), this is an estimate of the recovery R of activation patching,
on the same scale (1 = the node alone restores the sarcastic answer, 0 = no effect).
"""
import numpy as np
import torch

from .hooks import get_layers, hooks, run_with_cache
from .metric import d_from_logits, last_logits, yn_ids
from .model import model_info
from .patching import head_patching, score_pairs
from .prompts import REGIONS


def _corrupted_with_grads(model, tok, ids, kinds=("z", "mlp")):
    """Corrupted run that keeps every activation of `kinds` and its gradient w.r.t. d."""
    Y, N = yn_ids(tok)
    store = {}

    def keep(kind, l):
        def fn(t):
            t.retain_grad()
            store[(kind, l)] = t
            return t
        return fn

    n = len(get_layers(model))
    specs = [(k, l, keep(k, l)) for k in kinds for l in range(n)]
    # gradients must flow from the input, while the weights stay frozen
    emb = model.get_input_embeddings()
    h_emb = emb.register_forward_hook(lambda m, a, o: o.detach().requires_grad_(True))
    try:
        with torch.enable_grad(), hooks(model, specs):
            x = torch.tensor([ids], device=model.device)
            _, logits = last_logits(model, x)
            d = (torch.logsumexp(logits.float()[:, Y], -1) - torch.logsumexp(logits.float()[:, N], -1))[0]
            d.backward()
    finally:
        h_emb.remove()
    acts = {k: v.detach()[0] for k, v in store.items()}
    grads = {k: v.grad.detach()[0] for k, v in store.items()}
    return d.item(), acts, grads


def node_attribution(model, tok, pair):
    """Estimated R for every head and MLP layer, per region of the prompt.

    Returns dict:
        heads [n_layers, n_heads, n_regions]   estimated R of patching that head's z in that region
        mlp   [n_layers, n_regions]            estimated R of patching that MLP output in that region
        d_clean, d_corrupted
    """
    info = model_info(model)
    L, H, hd = info["n_layers"], info["n_heads"], info["head_dim"]
    model.requires_grad_(False)
    if "margin" not in pair.info:
        score_pairs(model, tok, [pair])
    _, clean = run_with_cache(model, pair.clean.ids, kinds=("z", "mlp"))
    _, corr, grad = _corrupted_with_grads(model, tok, pair.corrupted.ids)
    denom = pair.info["d_clean"] - pair.info["d_corrupted"]
    T = len(pair.corrupted.ids)
    regions = [pair.clean.regions[r] for r in REGIONS]

    heads = np.zeros((L, H, len(REGIONS)))
    mlp = np.zeros((L, len(REGIONS)))
    for l in range(L):
        diff = (clean[("z", l)].float() - corr[("z", l)].float()) * grad[("z", l)].float()   # [T, H*hd]
        per_head = diff.view(T, H, hd).sum(-1).cpu().numpy()                                  # [T, H]
        diff_m = ((clean[("mlp", l)].float() - corr[("mlp", l)].float()) * grad[("mlp", l)].float()).sum(-1).cpu().numpy()  # [T]
        for j, pos in enumerate(regions):
            heads[l, :, j] = per_head[pos].sum(0) / denom
            mlp[l, j] = diff_m[pos].sum() / denom
    return dict(heads=heads, mlp=mlp, d_clean=pair.info["d_clean"], d_corrupted=pair.info["d_corrupted"])


def attribution_over_pairs(model, tok, pairs, progress=True):
    """node_attribution for many pairs. Returns heads [n_pairs, L, H, R] and mlp [n_pairs, L, R]."""
    hs, ms = [], []
    for i, p in enumerate(pairs):
        a = node_attribution(model, tok, p)
        hs.append(a["heads"]); ms.append(a["mlp"])
        if progress and (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return dict(heads=np.stack(hs), mlp=np.stack(ms))


def actual_head_R(model, tok, pair):
    """Ground truth: real activation patching of every head in every region. Returns [L, H, n_regions]."""
    _, cache = run_with_cache(model, pair.clean.ids, kinds=("z",))
    out = [head_patching(model, tok, pair, pair.clean.regions[r], cache=cache) for r in REGIONS]
    return np.stack(out, axis=-1)


def top_heads(mean_heads, k=20):
    """The k (layer, head, region) entries with the largest mean estimated R."""
    flat = np.argsort(mean_heads, axis=None)[::-1][:k]
    L, H, Rn = mean_heads.shape
    rows = []
    for f in flat:
        l, h, r = np.unravel_index(f, (L, H, Rn))
        rows.append(dict(layer=int(l), head=int(h), region=REGIONS[r], est_R=float(mean_heads[l, h, r])))
    return rows
