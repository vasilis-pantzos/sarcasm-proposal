"""Edge attribution patching with integrated gradients (EAP-IG) and edge-level patching.

The graph
---------
Senders (they WRITE to the residual stream):
    input         the token embeddings
    a{l}.h{h}     attention head h of layer l (its output through o_proj)
    m{l}          the MLP of layer l
Receivers (they READ from the residual stream, at the input of their LayerNorm):
    a{l}          the attention block of layer l (all its heads read the same input)
    m{l}          the MLP block of layer l
    logits        the final norm, i.e. the answer
An edge u -> v exists if u writes before v reads: heads of layer l feed m{l} and everything after.
Receivers are whole blocks (not single heads); which head of a block reads an edge is studied later
with path patching (Phase 3).

EAP-IG score of edge u -> v (Hanna et al., 2024)
------------------------------------------------
    score(u, v) = sum_t (out_u^clean[t] - out_u^corr[t]) . mean_alpha  d(d)/d(in_v)[t]  at input e_alpha
where e_alpha = e_corr + alpha (e_clean - e_corr) interpolates the input embeddings (alpha = 1/m, ..., 1)
and in_v is the input of v's LayerNorm, taken only along v's branch. Divided by d(S) - d(corrupted),
the score is on the same scale as the recovery R.

Edge patching (to evaluate a circuit)
-------------------------------------
Run the clean input; every receiver v reads
    in_v = in_v^corr + sum_{u in circuit, u -> v} (out_u^now - out_u^corr)
so a circuit with all edges gives the clean answer and a circuit with no edges the corrupted one.
Faithfulness F = (d_circuit - d_corr) / (d_clean - d_corr).
"""
from dataclasses import dataclass

import numpy as np
import torch

from .metric import d_from_hidden, final_hidden, yn_ids
from .model import model_info


# ------------------------------------------------------------------ the graph
@dataclass
class Graph:
    senders: list          # names
    receivers: list        # names
    valid: np.ndarray      # [S, V] bool, edge exists
    n_layers: int
    n_heads: int

    def sender_index(self, name):
        return self.senders.index(name)

    def receiver_index(self, name):
        return self.receivers.index(name)

    def head_slice(self, l):
        """Rows of the heads of layer l among the senders."""
        start = 1 + l * self.n_heads
        return slice(start, start + self.n_heads)

    def mlp_row(self, l):
        return 1 + self.n_layers * self.n_heads + l

    def recv_attn(self, l):
        return l

    def recv_mlp(self, l):
        return self.n_layers + l

    def recv_logits(self):
        return 2 * self.n_layers

    def n_edges(self):
        return int(self.valid.sum())


def build_graph(model):
    info = model_info(model)
    L, H = info["n_layers"], info["n_heads"]
    senders = ["input"] + [f"a{l}.h{h}" for l in range(L) for h in range(H)] + [f"m{l}" for l in range(L)]
    receivers = [f"a{l}" for l in range(L)] + [f"m{l}" for l in range(L)] + ["logits"]
    # order in the forward pass: input = -1, heads of layer l = 2l, MLP l = 2l + 1
    s_order = np.array([-1] + [2 * l for l in range(L) for _ in range(H)] + [2 * l + 1 for l in range(L)])
    # a receiver at order k reads everything written before it: attn l reads < 2l, MLP l reads < 2l + 1
    r_order = np.array([2 * l for l in range(L)] + [2 * l + 1 for l in range(L)] + [2 * L])
    valid = s_order[:, None] < r_order[None, :]
    return Graph(senders, receivers, valid, L, H)


# ------------------------------------------------------------------ hooks on the modules
def _layers(model):
    return model.model.layers


def _receiver_modules(model, g):
    """The LayerNorm modules whose INPUT is each receiver's input, in receiver order."""
    layers = _layers(model)
    return ([layers[l].input_layernorm for l in range(g.n_layers)]
            + [layers[l].post_attention_layernorm for l in range(g.n_layers)]
            + [model.model.norm])


def _o_weight(model, l):
    """o_proj weight of layer l as float32 [d_model, H*hd], dequantized once (4-bit) and cached on the module."""
    lin = _layers(model)[l].self_attn.o_proj
    w = getattr(lin, "_eap_w32", None)
    if w is None:
        W = lin.weight
        if getattr(W, "quant_state", None) is not None:          # bitsandbytes 4-bit
            import bitsandbytes.functional as bnbF
            w = bnbF.dequantize_4bit(W.data, W.quant_state)
        else:
            w = W.data
        w = w.float().contiguous()
        lin._eap_w32 = w
    return w


def _head_delta(model, l, dz, H, hd):
    """Per-head change of the residual-stream write, in float32: dz [T, H*hd] -> [H, T, d_model]."""
    W = _o_weight(model, l)
    T = dz.shape[0]
    return torch.einsum("thk,dhk->htd", dz.float().view(T, H, hd), W.view(W.shape[0], H, hd))


def _heads_write(model, l, dz, w, H, hd):
    """sum_h w[h] * (change of head h's write), in float32: dz [T, H*hd], w [H] -> [T, d_model]."""
    T = dz.shape[0]
    zz = (dz.float().view(T, H, hd) * w.float()[None, :, None]).reshape(T, H * hd)
    return zz @ _o_weight(model, l).T


class _Capture:
    """Hooks that record the sender outputs (and optionally the receiver inputs) of one forward pass."""

    def __init__(self, model, g, receivers=False):
        self.model, self.g, self.receivers = model, g, receivers
        self.emb, self.z, self.mlp, self.recv = None, {}, {}, {}
        self.handles = []

    def __enter__(self):
        m = self.model
        self.handles.append(m.get_input_embeddings().register_forward_hook(
            lambda mod, a, o: setattr(self, "emb", o.detach())))
        for l, layer in enumerate(_layers(m)):
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda mod, a, l=l: self.z.__setitem__(l, a[0].detach())))
            self.handles.append(layer.mlp.register_forward_hook(
                lambda mod, a, o, l=l: self.mlp.__setitem__(l, o.detach())))
        if self.receivers:
            for v, mod in enumerate(_receiver_modules(m, self.g)):
                self.handles.append(mod.register_forward_pre_hook(
                    lambda mod, a, v=v: self.recv.__setitem__(v, a[0].detach())))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def _run_capture(model, ids, g, receivers=False):
    with _Capture(model, g, receivers) as c:
        final_hidden(model, torch.tensor([ids], device=model.device))
    return c


# ------------------------------------------------------------------ EAP-IG scores
def eap_ig_scores(model, tok, pair, steps=5):
    """EAP-IG score of every edge for one pair, divided by d(S) - d(corrupted). Returns [S, V] (0 where no edge)."""
    g = build_graph(model)
    info = model_info(model)
    H, hd, L = info["n_heads"], info["head_dim"], info["n_layers"]
    Y, N = yn_ids(tok)
    model.requires_grad_(False)

    clean = _run_capture(model, pair.clean.ids, g)
    corr = _run_capture(model, pair.corrupted.ids, g)
    with torch.no_grad():
        dc = d_from_hidden(model, final_hidden(model, torch.tensor([pair.clean.ids], device=model.device)), Y, N).item()
        dk = d_from_hidden(model, final_hidden(model, torch.tensor([pair.corrupted.ids], device=model.device)), Y, N).item()
    denom = dc - dk

    # integrated gradients: one batch of `steps` interpolated inputs
    alphas = torch.arange(1, steps + 1, device=model.device, dtype=torch.float32) / steps
    e_corr, e_clean = corr.emb.float(), clean.emb.float()                 # [1, T, d]
    e_interp = (e_corr + alphas[:, None, None] * (e_clean - e_corr)).to(clean.emb.dtype)   # [steps, T, d]
    probes = {}

    def add_probe(v):
        def pre(mod, args):
            p = torch.zeros_like(args[0], requires_grad=True)
            probes[v] = p
            return (args[0] + p,) + tuple(args[1:])
        return pre

    handles = [model.get_input_embeddings().register_forward_hook(lambda mod, a, o: e_interp)]
    handles += [mod.register_forward_pre_hook(add_probe(v)) for v, mod in enumerate(_receiver_modules(model, g))]
    try:
        with torch.enable_grad():
            x = torch.tensor([pair.clean.ids] * steps, device=model.device)
            d = d_from_hidden(model, final_hidden(model, x), Y, N)
            d.sum().backward()
    finally:
        for h in handles:
            h.remove()
    G = torch.stack([probes[v].grad.float().mean(0) for v in range(len(g.receivers))])   # [V, T, d]

    scores = np.zeros((len(g.senders), len(g.receivers)))
    with torch.no_grad():
        delta = (e_clean - e_corr)[0]                                      # [T, d]
        scores[0] = torch.einsum("td,vtd->v", delta, G).cpu().numpy()
        for l in range(L):
            dh = _head_delta(model, l, clean.z[l][0].float() - corr.z[l][0].float(), H, hd)
            scores[g.head_slice(l)] = torch.einsum("htd,vtd->hv", dh, G).cpu().numpy()
            dm = (clean.mlp[l] - corr.mlp[l])[0].float()
            scores[g.mlp_row(l)] = torch.einsum("td,vtd->v", dm, G).cpu().numpy()
    scores = np.where(g.valid, scores / denom, 0.0)
    return scores, dict(d_clean=dc, d_corrupted=dk)


def eap_ig_over_pairs(model, tok, pairs, steps=5, progress=True):
    out = []
    for i, p in enumerate(pairs):
        out.append(eap_ig_scores(model, tok, p, steps)[0].astype(np.float32))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if progress and (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs")
    return np.stack(out)


# ------------------------------------------------------------------ edge patching
class CircuitEvaluator:
    """Evaluate circuits (sets of edges) on one pair with edge-level patching.

    The corrupted run is cached once; each call to d(mask) is one forward pass on the clean input.

    Numerics: on the 4-bit / float16 model, rebuilding a receiver's input from hundreds of sender
    changes accumulates rounding error. So all changes are computed in float32, and each receiver's
    input is rebuilt from the nearer end:
        few edges in the circuit   ->  in_v = in_v^corr + sum_{included} change
        most edges in the circuit  ->  in_v = in_v^now  - sum_{excluded} change
    (in_v^now, the input arriving in this run, already contains every sender's change.) Both are the
    same quantity in exact arithmetic; this way "all edges" gives exactly the clean run and "no edges"
    exactly the corrupted run.
    """

    def __init__(self, model, tok, pair, graph=None):
        self.model, self.tok, self.pair = model, tok, pair
        self.g = graph or build_graph(model)
        info = model_info(model)
        self.H, self.hd = info["n_heads"], info["head_dim"]
        self.Y, self.N = yn_ids(tok)
        self.corr = _run_capture(model, pair.corrupted.ids, self.g, receivers=True)
        self.n_valid = self.g.valid.sum(0)
        for l in range(self.g.n_layers):
            _o_weight(model, l)                                            # dequantize once
        with torch.no_grad():
            x = lambda ids: torch.tensor([ids], device=model.device)
            self.d_clean = d_from_hidden(model, final_hidden(model, x(pair.clean.ids)), self.Y, self.N).item()
            self.d_corr = d_from_hidden(model, final_hidden(model, x(pair.corrupted.ids)), self.Y, self.N).item()

    @torch.no_grad()
    def d(self, mask):
        """d of the clean input when only the edges in mask [S, V] (bool) carry clean information."""
        m, g, H, hd = self.model, self.g, self.H, self.hd
        mask_np = np.asarray(mask, dtype=bool) & g.valid
        n_in = mask_np.sum(0)
        mask = torch.as_tensor(mask_np, device=m.device)
        valid = torch.as_tensor(g.valid, device=m.device)
        now = {"emb": None, "dz": {}, "mlp": {}}

        def on_emb(mod, a, o):
            now["emb"] = o[0].float() - self.corr.emb[0].float()
        def on_z(l):
            def pre(mod, a):
                now["dz"][l] = a[0][0].float() - self.corr.z[l][0].float()
            return pre
        def on_mlp(l):
            def hook(mod, a, o):
                now["mlp"][l] = o[0].float() - self.corr.mlp[l][0].float()
            return hook
        def on_recv(v):
            def pre(mod, a):
                if 2 * n_in[v] <= self.n_valid[v]:     # few edges: start from corrupted, add included
                    col, sign = mask[:, v], 1.0
                    x = self.corr.recv[v][0].float()
                else:                                  # most edges: start from now, remove excluded
                    col, sign = valid[:, v] & ~mask[:, v], -1.0
                    x = a[0][0].float()
                if col.any():
                    if col[0]:
                        x = x + sign * now["emb"]
                    for l, dz in now["dz"].items():
                        w = col[g.head_slice(l)]
                        if w.any():
                            x = x + sign * _heads_write(m, l, dz, w, H, hd)
                    for l, dm in now["mlp"].items():
                        if col[g.mlp_row(l)]:
                            x = x + sign * dm
                return (x.unsqueeze(0).to(a[0].dtype),) + tuple(a[1:])
            return pre

        layers = _layers(m)
        hs = [m.get_input_embeddings().register_forward_hook(on_emb)]
        hs += [layers[l].self_attn.o_proj.register_forward_pre_hook(on_z(l)) for l in range(g.n_layers)]
        hs += [layers[l].mlp.register_forward_hook(on_mlp(l)) for l in range(g.n_layers)]
        hs += [mod.register_forward_pre_hook(on_recv(v)) for v, mod in enumerate(_receiver_modules(m, g))]
        try:
            h = final_hidden(m, torch.tensor([self.pair.clean.ids], device=m.device))
        finally:
            for x in hs:
                x.remove()
        return d_from_hidden(m, h, self.Y, self.N).item()
    def faithfulness(self, mask):
        return (self.d(mask) - self.d_corr) / (self.d_clean - self.d_corr)


# ------------------------------------------------------------------ selecting circuits
def top_k_mask(scores, g, k, by_abs=True):
    """Boolean mask of the k highest-scoring valid edges."""
    s = np.abs(scores) if by_abs else scores.copy()
    s = np.where(g.valid, s, -np.inf)
    idx = np.argsort(s, axis=None)[::-1][:k]
    mask = np.zeros_like(g.valid)
    mask.flat[idx] = True
    return mask


def prune_dangling(mask, g):
    """Drop edges that cannot matter: their sender gets no clean information from the input through the
    circuit, or their receiver has no path to the logits through the circuit. Such edges carry exactly
    zero change, so the faithfulness is the same with fewer edges."""
    L, H = g.n_layers, g.n_heads
    m = np.asarray(mask, dtype=bool) & g.valid
    S, V = m.shape
    # forward: which senders receive clean information
    s_on = np.zeros(S, bool)
    s_on[0] = True
    for l in range(L):
        s_on[g.head_slice(l)] = m[s_on, g.recv_attn(l)].any()
        s_on[g.mlp_row(l)] = m[s_on, g.recv_mlp(l)].any()
    # backward: which receivers reach the logits
    r_use = np.zeros(V, bool)
    r_use[g.recv_logits()] = True
    for l in reversed(range(L)):
        r_use[g.recv_mlp(l)] = m[g.mlp_row(l), r_use].any()
        r_use[g.recv_attn(l)] = m[g.head_slice(l)][:, r_use].any()
    return m & s_on[:, None] & r_use[None, :]


def random_mask(g, k, rng):
    idx = rng.choice(np.flatnonzero(g.valid), size=k, replace=False)
    mask = np.zeros_like(g.valid)
    mask.flat[idx] = True
    return mask


def edge_table(scores_by_exp, g, mask=None):
    """Rows (sender, receiver, score per experiment, combined) for the edges in mask (or all valid)."""
    import pandas as pd
    sel = g.valid if mask is None else (mask & g.valid)
    si, vi = np.nonzero(sel)
    df = pd.DataFrame({"sender": [g.senders[i] for i in si], "receiver": [g.receivers[j] for j in vi]})
    for e, s in scores_by_exp.items():
        df[f"score_{e}"] = s[si, vi]
    df["combined"] = np.mean([s[si, vi] for s in scores_by_exp.values()], axis=0)
    return df.sort_values("combined", key=np.abs, ascending=False).reset_index(drop=True)
