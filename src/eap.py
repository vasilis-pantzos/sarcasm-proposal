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


def _head_outputs(model, l, z, H, hd):
    """Per-head contribution to the residual stream: z [B, T, H*hd] -> [B, H, T, d_model] (o_proj is linear)."""
    B, T, _ = z.shape
    mask = torch.zeros(H, H * hd, device=z.device, dtype=z.dtype)
    for h in range(H):
        mask[h, h * hd:(h + 1) * hd] = 1
    zz = (z.unsqueeze(1) * mask[None, :, None, :]).reshape(B * H, T, H * hd)
    out = _layers(model)[l].self_attn.o_proj(zz)
    return out.reshape(B, H, T, -1)


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
            dh = (_head_outputs(model, l, clean.z[l], H, hd) - _head_outputs(model, l, corr.z[l], H, hd))[0].float()
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
    """

    def __init__(self, model, tok, pair, graph=None):
        self.model, self.tok, self.pair = model, tok, pair
        self.g = graph or build_graph(model)
        info = model_info(model)
        self.H, self.hd = info["n_heads"], info["head_dim"]
        self.Y, self.N = yn_ids(tok)
        self.corr = _run_capture(model, pair.corrupted.ids, self.g, receivers=True)
        self.corr_heads = {l: _head_outputs(model, l, self.corr.z[l], self.H, self.hd)[0]
                           for l in range(self.g.n_layers)}                  # [H, T, d]
        with torch.no_grad():
            x = lambda ids: torch.tensor([ids], device=model.device)
            self.d_clean = d_from_hidden(model, final_hidden(model, x(pair.clean.ids)), self.Y, self.N).item()
            self.d_corr = d_from_hidden(model, final_hidden(model, x(pair.corrupted.ids)), self.Y, self.N).item()

    @torch.no_grad()
    def d(self, mask):
        """d of the clean input when only the edges in mask [S, V] (bool) carry clean information."""
        m, g = self.model, self.g
        mask = torch.as_tensor(np.asarray(mask) & g.valid, device=m.device)
        now = {"emb": None, "heads": {}, "mlp": {}}

        def on_emb(mod, a, o):
            now["emb"] = o
        busy = set()   # o_proj is called inside its own hook: do not recurse

        def on_z(l):
            def pre(mod, a):
                if l in busy:
                    return None
                busy.add(l)
                try:
                    now["heads"][l] = _head_outputs(m, l, a[0], self.H, self.hd)[0] - self.corr_heads[l]
                finally:
                    busy.discard(l)
            return pre
        def on_mlp(l):
            def hook(mod, a, o):
                now["mlp"][l] = o[0] - self.corr.mlp[l][0]
            return hook
        def on_recv(v):
            def pre(mod, a):
                col = mask[:, v]
                x = self.corr.recv[v][0].clone()
                if col[0]:
                    x = x + (now["emb"][0] - self.corr.emb[0])
                for l, dh in now["heads"].items():
                    w = col[g.head_slice(l)].to(dh.dtype)
                    if w.any():
                        x = x + torch.einsum("h,htd->td", w, dh)
                for l, dm in now["mlp"].items():
                    if col[g.mlp_row(l)]:
                        x = x + dm
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
