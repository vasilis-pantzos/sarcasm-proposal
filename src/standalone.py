"""Circuits that answer on their own: edge-level MEAN ablation on single prompts.

Edge patching in eap.py / acdc.py needs a corrupted partner for every prompt. To let a circuit answer a
single prompt like a normal model, the edges NOT in the circuit get the dataset MEAN instead of a corrupted
run (as in IOI / ACDC "mean ablation"):

    receiver input = mean_input + sum_{u in circuit} (output_u(prompt) - mean_output_u)

All edges -> exactly the full model; no edges -> the "average prompt" (the same answer for every prompt).

The means are taken position by position over prompts of the SAME number of tokens (all our prompts share the
chat template; only the situation + phrase in the middle differ). A prompt whose length is not in the
reference gets the closest length, with the template prefix and suffix aligned and the middle resampled.
"""
import numpy as np
import torch

from .acdc import BatchEvaluator, _o_weight_native
from .eap import _Capture, build_graph
from .metric import d_from_hidden, final_hidden, yn_ids
from .model import model_info


class _Ref:
    """Same attributes as eap._Capture: emb, z[l], mlp[l], recv[v] (each [1, T, dim])."""

    def __init__(self, emb, z, mlp, recv):
        self.emb, self.z, self.mlp, self.recv = emb, z, mlp, recv

    def expand(self, B):
        e = lambda t: t.expand(B, -1, -1)
        return _Ref(e(self.emb), {k: e(v) for k, v in self.z.items()}, {k: e(v) for k, v in self.mlp.items()},
                    {k: e(v) for k, v in self.recv.items()})

    def take(self, idx):
        t = lambda x: x[:, idx]
        return _Ref(t(self.emb), {k: t(v) for k, v in self.z.items()}, {k: t(v) for k, v in self.mlp.items()},
                    {k: t(v) for k, v in self.recv.items()})


class MeanReference:
    """Mean activations (sender outputs and receiver inputs) per prompt length."""

    @torch.no_grad()
    def __init__(self, model, id_lists, graph=None, prefix=None, suffix=None, batch=32):
        """prefix / suffix: number of template tokens before / after the variable middle (for other lengths)."""
        self.g = graph or build_graph(model)
        self.prefix, self.suffix = prefix, suffix
        by_len = {}
        for ids in id_lists:
            by_len.setdefault(len(ids), []).append(ids)
        self.means, self.counts = {}, {}
        for T, group in sorted(by_len.items()):
            acc, n = None, 0
            for i in range(0, len(group), batch):
                x = torch.tensor(group[i:i + batch], device=model.device)
                with _Capture(model, self.g, receivers=True) as c:
                    final_hidden(model, x)
                parts = [c.emb] + [c.z[l] for l in range(self.g.n_layers)] + [c.mlp[l] for l in range(self.g.n_layers)] \
                    + [c.recv[v] for v in range(len(self.g.receivers))]
                s = [p.float().sum(0, keepdim=True) for p in parts]
                acc = s if acc is None else [a + b for a, b in zip(acc, s)]
                n += x.shape[0]
            dt = parts[0].dtype
            m = [(a / n).to(dt) for a in acc]
            L, V = self.g.n_layers, len(self.g.receivers)
            self.means[T] = _Ref(m[0], {l: m[1 + l] for l in range(L)}, {l: m[1 + L + l] for l in range(L)},
                                 {v: m[1 + 2 * L + v] for v in range(V)})
            self.counts[T] = n

    def for_length(self, T):
        """The mean for prompts of T tokens (exact if that length was seen, else aligned from the closest)."""
        if T in self.means:
            return self.means[T], True
        T2 = min(self.means, key=lambda t: abs(t - T))
        P, S = self.prefix, self.suffix
        assert P is not None and S is not None, "prefix/suffix needed for an unseen length"
        mid, mid2 = T - P - S, T2 - P - S
        idx = list(range(P)) + [P + int(i * mid2 / max(mid, 1)) for i in range(mid)] + list(range(T2 - S, T2))
        return self.means[T2].take(torch.tensor(idx, device=self.means[T2].emb.device)), False


class StandaloneEvaluator(BatchEvaluator):
    """BatchEvaluator whose 'corrupted' run is the mean reference: prompts of one length, no partner."""

    def __init__(self, model, tok, id_lists, ref, graph=None):
        self.model, self.tok = model, tok
        self.g = graph or build_graph(model)
        info = model_info(model)
        self.H, self.hd = info["n_heads"], info["head_dim"]
        self.Y, self.N = yn_ids(tok)
        self.loss_mode = "mean"
        T = len(id_lists[0])
        assert all(len(x) == T for x in id_lists), "one length per evaluator"
        self.clean = torch.tensor(id_lists, device=model.device)
        self.kw = {}
        r, self.exact_mean = ref.for_length(T)
        self.corr = r.expand(len(id_lists))
        self.n_valid = self.g.valid.sum(0)
        self.W = [_o_weight_native(model, l) for l in range(self.g.n_layers)]
        self.n_calls = 0
        with torch.no_grad():
            self.d_clean = d_from_hidden(model, final_hidden(model, self.clean), self.Y, self.N).cpu().numpy()


def standalone_d(model, tok, id_lists, masks, ref, graph=None, batch=32, progress=True):
    """d of every prompt for the full model and for every circuit in {name: mask}.

    Returns {"full": [n], name: [n], ...} in the order of id_lists."""
    g = graph or build_graph(model)
    out = {k: np.zeros(len(id_lists)) for k in ["full", *masks]}
    by_len = {}
    for i, ids in enumerate(id_lists):
        by_len.setdefault(len(ids), []).append(i)
    done = 0
    for T, idx in sorted(by_len.items()):
        for j in range(0, len(idx), batch):
            part = idx[j:j + batch]
            ev = StandaloneEvaluator(model, tok, [id_lists[i] for i in part], ref, g)
            out["full"][part] = ev.d_clean
            for k, m in masks.items():
                out[k][part] = ev.d(m)
            del ev
            done += len(part)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if progress:
            print(f"  length {T}: {len(idx)} prompts ({done}/{len(id_lists)})")
    return out
