"""ACDC (Conmy et al., 2023) on the edge graph of eap.py: find a minimal circuit by removing edges.

Start from a candidate circuit (the EAP-IG edges of Phase 1b). Go through the receivers from the
answer backwards (logits, m27, a27, m26, ...). For each receiver, try to remove its incoming edges;
an edge is removed for good if the circuit still explains the behaviour:

    loss(circuit) = max over experiments of |1 - mean F|        (F = faithfulness, 1 = like the full model)
    remove if  loss(without) - loss(with) < tau  and  loss(without) <= cap

Two changes make it fast enough for the 7B model on a T4:
  * batched evaluation: all pairs of both experiments go through ONE forward pass (BatchEvaluator);
  * hierarchical removal: first try to remove ALL incoming edges of a receiver at once; if that is
    rejected, split them in two halves (ordered by layer) and recurse, down to single edges.
    Unimportant groups are removed with one test; single edges are tested only where it matters.
Receivers that no longer reach the answer lose all their incoming edges without a test (prune_dangling).
The state is saved after every receiver (and every 100 tests), so an interrupted run continues.
"""
import os
import time

import numpy as np
import torch

from .eap import _Capture, _layers, _receiver_modules, build_graph, prune_dangling
from .metric import d_from_hidden, final_hidden, yn_ids
from .model import model_info


def _o_weight_native(model, l):
    """o_proj weight of layer l in the model's compute dtype [d_model, H*hd], dequantized once and cached."""
    lin = _layers(model)[l].self_attn.o_proj
    w = getattr(lin, "_eap_wn", None)
    if w is None:
        W = lin.weight
        if getattr(W, "quant_state", None) is not None:          # bitsandbytes 4-bit
            import bitsandbytes.functional as bnbF
            w = bnbF.dequantize_4bit(W.data, W.quant_state)
        else:
            w = W.data
        w = w.to(model.get_input_embeddings().weight.dtype).contiguous()
        lin._eap_wn = w
    return w


def _pad(pairs, pad_id, device):
    """Left-pad the clean and corrupted ids of all pairs to one length."""
    T = max(len(p.clean.ids) for p in pairs)
    B = len(pairs)
    clean = torch.full((B, T), pad_id, dtype=torch.long)
    corr = torch.full((B, T), pad_id, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.long)
    for b, p in enumerate(pairs):
        assert len(p.clean.ids) == len(p.corrupted.ids), "pairs must be token-matched"
        n = len(p.clean.ids)
        clean[b, T - n:] = torch.tensor(p.clean.ids)
        corr[b, T - n:] = torch.tensor(p.corrupted.ids)
        mask[b, T - n:] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    to = lambda x: x.to(device)
    return to(clean), to(corr), dict(attention_mask=to(mask), position_ids=to(pos))


class BatchEvaluator:
    """Edge patching for many pairs at once: d(mask) is ONE forward pass over all pairs.

    Same method as eap.CircuitEvaluator (changes in float32, receiver inputs rebuilt from the nearer
    end, so all edges = clean and no edges = corrupted exactly); the head writes use one matrix
    product with the dequantized o_proj weight in the model's dtype, accumulated into float32.
    """

    def __init__(self, model, tok, pairs, graph=None, loss="mean"):
        """loss: "mean"     -> max over experiments of |1 - mean F|            (ACDC v1)
                 "per_pair" -> max over experiments of mean over pairs |1 - F|  (ACDC v2: a pair that
                               breaks cannot hide behind a pair that overshoots)"""
        assert loss in ("mean", "per_pair")
        self.loss_mode = loss
        self.model, self.tok, self.pairs = model, tok, list(pairs)
        self.g = graph or build_graph(model)
        info = model_info(model)
        self.H, self.hd = info["n_heads"], info["head_dim"]
        self.Y, self.N = yn_ids(tok)
        self.exp = np.array([p.experiment for p in self.pairs])
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.clean, corr_ids, self.kw = _pad(self.pairs, pad, model.device)
        with torch.no_grad(), _Capture(model, self.g, receivers=True) as c:
            self.d_corr = d_from_hidden(model, final_hidden(model, corr_ids, **self.kw), self.Y, self.N).cpu().numpy()
        self.corr = c
        with torch.no_grad():
            self.d_clean = d_from_hidden(model, final_hidden(model, self.clean, **self.kw), self.Y, self.N).cpu().numpy()
        self.n_valid = self.g.valid.sum(0)
        self.W = [_o_weight_native(model, l) for l in range(self.g.n_layers)]
        self.n_calls = 0

    def _heads_write(self, l, dz, w):
        B, T, _ = dz.shape
        zz = (dz.view(B, T, self.H, self.hd) * w.to(dz.dtype)[None, None, :, None]).reshape(B, T, -1)
        W = self.W[l]
        return (zz.to(W.dtype) @ W.T).float()

    @torch.no_grad()
    def d(self, mask):
        """d of every pair [n_pairs] when only the edges in mask [S, V] carry clean information."""
        m, g = self.model, self.g
        mask_np = np.asarray(mask, dtype=bool) & g.valid
        n_in = mask_np.sum(0)
        mask = torch.as_tensor(mask_np, device=m.device)
        valid = torch.as_tensor(g.valid, device=m.device)
        now = {"emb": None, "dz": {}, "mlp": {}}
        c = self.corr

        def on_emb(mod, a, o):
            now["emb"] = o.float() - c.emb.float()
        def on_z(l):
            def pre(mod, a):
                now["dz"][l] = a[0].float() - c.z[l].float()
            return pre
        def on_mlp(l):
            def hook(mod, a, o):
                now["mlp"][l] = o.float() - c.mlp[l].float()
            return hook
        def on_recv(v):
            def pre(mod, a):
                if 2 * n_in[v] <= self.n_valid[v]:
                    col, sign, x = mask[:, v], 1.0, c.recv[v].float()
                else:
                    col, sign, x = valid[:, v] & ~mask[:, v], -1.0, a[0].float()
                if col.any():
                    if col[0]:
                        x = x + sign * now["emb"]
                    for l, dz in now["dz"].items():
                        w = col[g.head_slice(l)]
                        if w.any():
                            x = x + sign * self._heads_write(l, dz, w)
                    for l, dm in now["mlp"].items():
                        if col[g.mlp_row(l)]:
                            x = x + sign * dm
                return (x.to(a[0].dtype),) + tuple(a[1:])
            return pre

        layers = _layers(m)
        hs = [m.get_input_embeddings().register_forward_hook(on_emb)]
        hs += [layers[l].self_attn.o_proj.register_forward_pre_hook(on_z(l)) for l in range(g.n_layers)]
        hs += [layers[l].mlp.register_forward_hook(on_mlp(l)) for l in range(g.n_layers)]
        hs += [mod.register_forward_pre_hook(on_recv(v)) for v, mod in enumerate(_receiver_modules(m, g))]
        try:
            h = final_hidden(m, self.clean, **self.kw)
        finally:
            for x in hs:
                x.remove()
        self.n_calls += 1
        return d_from_hidden(m, h, self.Y, self.N).cpu().numpy()

    def faithfulness(self, mask):
        """F of every pair [n_pairs]."""
        return (self.d(mask) - self.d_corr) / (self.d_clean - self.d_corr)

    def mean_F(self, mask):
        """Mean F per experiment: {experiment: F}."""
        F = self.faithfulness(mask)
        return {e: float(F[self.exp == e].mean()) for e in np.unique(self.exp)}

    def loss(self, mask):
        F = self.faithfulness(mask)
        if self.loss_mode == "mean":
            return max(abs(1 - F[self.exp == e].mean()) for e in np.unique(self.exp))
        return max(np.abs(1 - F[self.exp == e]).mean() for e in np.unique(self.exp))


# ------------------------------------------------------------------ ACDC
def receiver_order(g):
    """Receivers from the answer backwards: logits, m{L-1}, a{L-1}, ..., m0, a0."""
    out = [g.recv_logits()]
    for l in reversed(range(g.n_layers)):
        out += [g.recv_mlp(l), g.recv_attn(l)]
    return out


def _sender_layer(g, s):
    if s == 0:
        return -1.0
    if s < g.mlp_row(0):
        return float((s - 1) // g.n_heads)
    return s - g.mlp_row(0) + 0.5


def acdc(ev, start_mask, tau, cap, state_path=None, scores=None, progress=True, monitor=None, monitor_every=4):
    """Hierarchical ACDC. Returns dict(mask, loss, n_tests, log); log rows are
    (receiver, n_edges_tested, loss_before, loss_after, removed).

    scores [S, V] (optional, e.g. the EAP-IG scores): the incoming edges of a receiver are ordered by
    |score| before splitting, so the weak edges end up together and go with few tests. Without
    scores they are ordered by layer. On resume the saved cap is used.

    monitor (optional): function mask -> {name: value}, e.g. the mean F on pairs ACDC does not see.
    It is called every `monitor_every` receivers; the values are printed and kept in the result,
    so overfitting shows up during the run."""
    g = ev.g
    if state_path and os.path.exists(state_path):
        st = torch.load(state_path, weights_only=False)
        print(f"resuming: {len(st['done'])}/{len(g.receivers)} receivers done, {int(st['mask'].sum())} edges")
    else:
        st = dict(mask=prune_dangling(start_mask, g), done=[], log=[], n_tests=0, tau=tau, cap=cap, monitor=[])
    assert st["tau"] == tau, "saved state has a different tau: delete it or change the name"
    cap = st["cap"]
    mask = st["mask"]
    cur = ev.loss(mask)
    t0 = time.time()

    def save():
        if state_path:
            st["mask"] = mask
            torch.save(st, state_path)

    for v in receiver_order(g):
        if v in st["done"]:
            continue
        n_before = int(mask[:, v].sum())
        tests_before = st["n_tests"]
        mask = prune_dangling(mask, g)                # edges into a receiver that reaches nothing: no test needed
        if scores is None:
            edges = sorted(np.flatnonzero(mask[:, v]), key=lambda s: (_sender_layer(g, s), s))
        else:
            edges = sorted(np.flatnonzero(mask[:, v]), key=lambda s: abs(scores[s, v]))   # weakest first

        def process(group, known=None):
            """Try to remove all edges of `group` into v; if rejected, split in two and recurse.
            `known`: the loss without the group, if already measured. Returns True if removed whole."""
            nonlocal mask, cur
            trial = mask.copy()
            trial[group, v] = False
            if known is None:
                new = ev.loss(trial)
                st["n_tests"] += 1
                if st["n_tests"] % 100 == 0:
                    save()
            else:
                new = known
            ok = (new - cur < tau) and (new <= cap)
            st["log"].append((g.receivers[v], len(group), cur, new, ok))
            if ok:
                mask, cur = trial, new
                return True
            if len(group) > 1:
                half = len(group) // 2
                first_removed = process(group[:half])
                # if the first half went, removing the second half gives the circuit we already measured
                process(group[half:], known=new if first_removed else None)
            return False

        if edges:
            process(edges)
        st["done"].append(v)
        save()
        if progress:
            print(f"  {g.receivers[v]:>6s}: {n_before:5d} -> {int(mask[:, v].sum()):4d} edges in, "
                  f"{st['n_tests'] - tests_before:4d} tests | circuit {int(mask.sum()):5d} edges, "
                  f"loss {cur:.3f} | {(time.time() - t0) / 60:.1f} min")
        if monitor is not None and len(st["done"]) % monitor_every == 0:
            vals = monitor(prune_dangling(mask, g))
            st.setdefault("monitor", []).append((len(st["done"]), int(mask.sum()), vals))
            save()
            print("          monitor:", {k: round(float(x), 3) for k, x in vals.items()})
    mask = prune_dangling(mask, g)
    st["mask"] = mask
    save()
    return dict(mask=mask, loss=ev.loss(mask), n_tests=st["n_tests"], log=st["log"], monitor=st.get("monitor", []))


# ------------------------------------------------------------------ evaluation of circuits
def faithfulness_many(model, tok, pairs, masks, graph=None, batch=16):
    """F of every pair for every mask in {name: mask}. Returns {name: array [n_pairs]}."""
    out = {k: [] for k in masks}
    for i in range(0, len(pairs), batch):
        ev = BatchEvaluator(model, tok, pairs[i:i + batch], graph)
        for k, m in masks.items():
            out[k].append(ev.faithfulness(m))
        del ev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {k: np.concatenate(v) for k, v in out.items()}


# ------------------------------------------------------------------ ACDC v3: one cut-off per receiver
class MultiEvaluator:
    """Several BatchEvaluators (one per chunk of pairs) behind the same interface: many pairs per test."""

    def __init__(self, model, tok, pairs, graph=None, loss="per_pair", batch=32):
        self.g = graph or build_graph(model)
        self.evs = [BatchEvaluator(model, tok, pairs[i:i + batch], self.g, loss=loss) for i in range(0, len(pairs), batch)]
        self.exp = np.concatenate([e.exp for e in self.evs])
        self.loss_mode = loss
        self.d_clean = np.concatenate([e.d_clean for e in self.evs])
        self.d_corr = np.concatenate([e.d_corr for e in self.evs])

    def faithfulness(self, mask):
        return np.concatenate([e.faithfulness(mask) for e in self.evs])

    def mean_F(self, mask):
        F = self.faithfulness(mask)
        return {e: float(F[self.exp == e].mean()) for e in np.unique(self.exp)}

    def loss(self, mask):
        F = self.faithfulness(mask)
        if self.loss_mode == "mean":
            return max(abs(1 - F[self.exp == e].mean()) for e in np.unique(self.exp))
        return max(np.abs(1 - F[self.exp == e]).mean() for e in np.unique(self.exp))


def acdc_cutoff(ev, start_mask, scores, tau, cap, state_path=None, progress=True, monitor=None, monitor_every=4):
    """ACDC with ONE decision per receiver: how many of its strongest incoming edges (by |EAP-IG score|) to keep.

    For each receiver (from the answer backwards) the incoming edges are ranked by |score|; we look for the
    smallest k such that keeping only the top-k still satisfies
        loss(top-k) - loss(current) < tau   and   loss(top-k) <= cap,
    by trying k = 0 first and then a binary search on k (about log2(n) tests per receiver). The ranking comes
    from EAP-IG on all discovery pairs, so ACDC only chooses ~one number per receiver instead of thousands of
    edge-by-edge decisions: much less room to overfit the pairs it is tested on.
    """
    g = ev.g
    if state_path and os.path.exists(state_path):
        st = torch.load(state_path, weights_only=False)
        print(f"resuming: {len(st['done'])}/{len(g.receivers)} receivers done, {int(st['mask'].sum())} edges")
    else:
        st = dict(mask=prune_dangling(start_mask, g), done=[], log=[], n_tests=0, tau=tau, cap=cap, monitor=[], kept={})
    assert st["tau"] == tau, "saved state has a different tau: delete it or change the name"
    cap = st["cap"]
    mask = st["mask"]
    cur = ev.loss(mask)
    t0 = time.time()

    def save():
        if state_path:
            st["mask"] = mask
            torch.save(st, state_path)

    for v in receiver_order(g):
        if v in st["done"]:
            continue
        tests_before = st["n_tests"]
        mask = prune_dangling(mask, g)
        edges = sorted(np.flatnonzero(mask[:, v]), key=lambda s: -abs(scores[s, v]))    # strongest first
        n = len(edges)
        losses = {n: cur}

        def trial_mask(k):
            m = mask.copy()
            m[edges[k:], v] = False
            return m

        def ok(k):
            if k not in losses:
                losses[k] = ev.loss(trial_mask(k))
                st["n_tests"] += 1
            new = losses[k]
            st["log"].append((g.receivers[v], n, k, cur, new, (new - cur < tau) and (new <= cap)))
            return (new - cur < tau) and (new <= cap)

        if n == 0:
            k = 0
        elif ok(0):
            k = 0
        else:
            lo, hi = 0, n                     # ok(hi) is true (the current circuit), ok(lo) is false
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if ok(mid):
                    hi = mid
                else:
                    lo = mid
            k = hi
        mask = trial_mask(k)
        cur = losses[k]
        st["kept"][g.receivers[v]] = (k, n)
        st["done"].append(v)
        save()
        if progress:
            print(f"  {g.receivers[v]:>6s}: keep {k:4d} of {n:4d} incoming edges, {st['n_tests'] - tests_before:3d} tests | "
                  f"circuit {int(prune_dangling(mask, g).sum()):5d} edges, loss {cur:.3f} | {(time.time() - t0) / 60:.1f} min")
        if monitor is not None and len(st["done"]) % monitor_every == 0:
            vals = monitor(prune_dangling(mask, g))
            st["monitor"].append((len(st["done"]), int(prune_dangling(mask, g).sum()), vals))
            save()
            print("          monitor:", {kk: round(float(x), 3) for kk, x in vals.items()})
    mask = prune_dangling(mask, g)
    st["mask"] = mask
    save()
    return dict(mask=mask, loss=ev.loss(mask), n_tests=st["n_tests"], log=st["log"], monitor=st["monitor"], kept=st["kept"])
