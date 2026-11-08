"""Activation patching (denoising): run the corrupted input and put back clean activations.

Recovery R = (d_patched - d_corrupted) / (d_clean - d_corrupted)
    R = 1: the patched component alone restores the sarcastic answer; R = 0: no effect.
"""
import numpy as np
import torch

from .hooks import get_layers, hooks, replace_heads, replace_positions, run_with_cache
from .metric import d_of, d_single
from .model import model_info
from .prompts import REGIONS


def recovery(d_patched, d_clean, d_corr):
    return (np.asarray(d_patched) - d_corr) / (d_clean - d_corr)


def score_pairs(model, tok, pairs):
    """Fill pair.info with d_clean, d_corrupted and their difference (the margin)."""
    for p in pairs:
        p.info["d_clean"] = d_single(model, tok, p.clean.ids)
        p.info["d_corrupted"] = d_single(model, tok, p.corrupted.ids)
        p.info["margin"] = p.info["d_clean"] - p.info["d_corrupted"]
    return pairs


def keep_correct(pairs, min_margin=1.0):
    """Keep pairs the model answers correctly (margin >= min_margin), so that R is well defined."""
    return [p for p in pairs if p.info.get("margin", -np.inf) >= min_margin]


@torch.no_grad()
def sweep_positions(model, tok, pair, kind, pos_groups, cache=None):
    """For every layer: patch `kind` at each group of positions (one group per batch row).

    Returns d after patching, shape [n_layers, len(pos_groups)].
    """
    if cache is None:
        _, cache = run_with_cache(model, pair.clean.ids, kinds=(kind,))
    x = torch.tensor([pair.corrupted.ids] * len(pos_groups), device=model.device)
    out = []
    for l in range(len(get_layers(model))):
        with hooks(model, [(kind, l, replace_positions(cache[(kind, l)], pos_groups))]):
            out.append(d_of(model, tok, x))
    return torch.stack(out).numpy()


def region_patching(model, tok, pairs, kinds=("resid", "attn", "mlp"), progress=True):
    """Patch each kind, layer and region for every pair.

    Returns {kind: R array [n_pairs, n_layers, n_regions]} with regions in the order REGIONS.
    """
    res = {k: [] for k in kinds}
    for i, p in enumerate(pairs):
        if "margin" not in p.info:
            score_pairs(model, tok, [p])
        _, cache = run_with_cache(model, p.clean.ids, kinds=kinds)
        groups = [p.clean.regions[r] for r in REGIONS]
        for k in kinds:
            d = sweep_positions(model, tok, p, k, groups, cache)
            res[k].append(recovery(d, p.info["d_clean"], p.info["d_corrupted"]))
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if progress and (i + 1) % 4 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs")
    return {k: np.stack(v) for k, v in res.items()}


@torch.no_grad()
def head_patching(model, tok, pair, positions, layers=None, cache=None):
    """Patch each attention head separately (its z at `positions`).

    One batch per layer, one row per head. Returns R [n_layers, n_heads] (NaN for skipped layers).
    """
    info = model_info(model)
    H, hd, L = info["n_heads"], info["head_dim"], info["n_layers"]
    layers = range(L) if layers is None else layers
    if "margin" not in pair.info:
        score_pairs(model, tok, [pair])
    if cache is None:
        _, cache = run_with_cache(model, pair.clean.ids, kinds=("z",))
    x = torch.tensor([pair.corrupted.ids] * H, device=model.device)
    R = np.full((L, H), np.nan)
    for l in layers:
        fn = replace_heads(cache[("z", l)], [[h] for h in range(H)], positions, hd)
        with hooks(model, [("z", l, fn)]):
            d = d_of(model, tok, x).numpy()
        R[l] = recovery(d, pair.info["d_clean"], pair.info["d_corrupted"])
    return R


def mean_and_se(arr, axis=0):
    """Mean and standard error over pairs."""
    arr = np.asarray(arr)
    return arr.mean(axis), arr.std(axis, ddof=1) / np.sqrt(arr.shape[axis])
