"""Read and replace internal activations of a Qwen2-style model with PyTorch hooks.

Kinds of activation (each one per layer l):
    resid   residual stream at the INPUT of layer l              [batch, seq, d_model]
    attn    output of the attention block of layer l            [batch, seq, d_model]
    mlp     output of the MLP block of layer l                   [batch, seq, d_model]
    z       per-head outputs before the output projection o_proj [batch, seq, n_heads * head_dim]
            (head h is the slice h*head_dim:(h+1)*head_dim)
"""
from contextlib import contextmanager

import torch

from .metric import last_logits

KINDS = ("resid", "attn", "mlp", "z")


def get_layers(model):
    return model.model.layers


def _first(o):
    return o[0] if isinstance(o, tuple) else o


def _replace_first(o, new):
    return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new


def _register(layer, kind, fn):
    """Register a hook that maps the activation `kind` of `layer` through fn(tensor) -> tensor."""
    if kind == "resid":
        def pre(m, args, kwargs):
            if args:
                return (fn(args[0]),) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = fn(kwargs["hidden_states"])
            return args, kwargs
        return layer.register_forward_pre_hook(pre, with_kwargs=True)
    if kind == "attn":
        return layer.self_attn.register_forward_hook(lambda m, a, o: _replace_first(o, fn(_first(o))))
    if kind == "mlp":
        return layer.mlp.register_forward_hook(lambda m, a, o: fn(o))
    if kind == "z":
        def pre(m, args, kwargs):
            if args:
                return (fn(args[0]),) + tuple(args[1:]), kwargs
            kwargs["input"] = fn(kwargs["input"])
            return args, kwargs
        return layer.self_attn.o_proj.register_forward_pre_hook(pre, with_kwargs=True)
    raise ValueError(f"unknown kind {kind!r}, expected one of {KINDS}")


@contextmanager
def hooks(model, specs):
    """Temporarily apply hooks. specs: list of (kind, layer_index, fn)."""
    layers = get_layers(model)
    handles = [_register(layers[l], kind, fn) for kind, l, fn in specs]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def run_with_cache(model, ids, kinds=("resid", "attn", "mlp")):
    """Run one sequence and store its activations.

    Returns (logits of the last position [1, vocab], cache) where cache[(kind, l)] is a tensor [seq, dim].
    """
    cache = {}

    def saver(kind, l):
        def fn(t):
            cache[(kind, l)] = t[0].detach().clone()
            return t
        return fn

    n = len(get_layers(model))
    specs = [(k, l, saver(k, l)) for k in kinds for l in range(n)]
    x = torch.tensor([ids], device=model.device)
    with hooks(model, specs):
        _, logits = last_logits(model, x)
    return logits, cache


def replace_positions(clean, pos_groups):
    """fn for a batch where row b takes the clean values at the positions pos_groups[b]."""
    def fn(t):
        t = t.clone()
        for b, pos in enumerate(pos_groups):
            if len(pos):
                t[b, pos] = clean[pos].to(t.dtype)
        return t
    return fn


def replace_heads(clean, head_groups, positions, head_dim):
    """fn for z where row b takes the clean values of the heads head_groups[b] at `positions`."""
    def fn(t):
        t = t.clone()
        for b, heads in enumerate(head_groups):
            for h in heads:
                sl = slice(h * head_dim, (h + 1) * head_dim)
                t[b, positions, sl] = clean[positions, sl].to(t.dtype)
        return t
    return fn
