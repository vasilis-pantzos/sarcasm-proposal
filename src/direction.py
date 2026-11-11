"""Phase 3a: a 'sarcasm direction' in the residual stream of the full model.

* collect_resid: the residual stream at the LAST token, at the input of every layer (and after the last one),
  for many prompts, plus d of the full model;
* mean_diff / probe accuracy per layer (see the notebook);
* steer_d: add alpha * vector to the residual stream at the last token at the input of one layer, and read d;
* component_writes: how much each attention head and MLP writes along a direction (at the last token).

Prompts are processed in batches of equal length, so there is no padding and the last token is position -1.
"""
import numpy as np
import torch

from .eap import _layers
from .metric import d_from_hidden, final_hidden, yn_ids
from .model import model_info


def _by_length(id_lists, batch):
    groups = {}
    for i, ids in enumerate(id_lists):
        groups.setdefault(len(ids), []).append(i)
    for T, idx in sorted(groups.items()):
        for j in range(0, len(idx), batch):
            yield idx[j:j + batch]


@torch.no_grad()
def collect_resid(model, tok, id_lists, batch=32, progress=True):
    """Residual stream at the last token: X [n, n_layers + 1, d_model] (float16; X[:, l] = input of layer l,
    X[:, L] = output of the last layer before the final norm) and d of the full model [n]."""
    L = len(_layers(model))
    Y, N = yn_ids(tok)
    n, D = len(id_lists), model.config.hidden_size
    X = np.zeros((n, L + 1, D), dtype=np.float16)
    d = np.zeros(n, dtype=np.float32)
    store = {}
    hooks = [layer.register_forward_pre_hook(lambda m, a, l=l: store.__setitem__(l, a[0][:, -1].float().cpu()))
             for l, layer in enumerate(_layers(model))]
    hooks.append(model.model.norm.register_forward_pre_hook(lambda m, a: store.__setitem__(L, a[0][:, -1].float().cpu())))
    try:
        done = 0
        for idx in _by_length(id_lists, batch):
            x = torch.tensor([id_lists[i] for i in idx], device=model.device)
            h = final_hidden(model, x)
            d[idx] = d_from_hidden(model, h, Y, N).cpu().numpy()
            X[idx] = torch.stack([store[l] for l in range(L + 1)], 1).numpy().astype(np.float16)
            done += len(idx)
            if progress and done % 320 < len(idx):
                print(f"  {done}/{n} prompts")
    finally:
        for hk in hooks:
            hk.remove()
    return X, d


@torch.no_grad()
def steer_d(model, tok, id_lists, layer, vec, alphas, batch=32):
    """d of every prompt when alpha * vec is added to the residual stream at the last token, at the input of
    `layer`. vec: [d_model]. Returns [len(alphas), n]."""
    Y, N = yn_ids(tok)
    v = torch.as_tensor(np.asarray(vec), dtype=torch.float32, device=model.device)
    out = np.zeros((len(alphas), len(id_lists)), dtype=np.float32)
    state = {"a": 0.0}

    def pre(m, a):
        h = a[0].clone()
        h[:, -1] += (state["a"] * v).to(h.dtype)
        return (h,) + tuple(a[1:])

    hk = _layers(model)[layer].register_forward_pre_hook(pre)
    try:
        for idx in _by_length(id_lists, batch):
            x = torch.tensor([id_lists[i] for i in idx], device=model.device)
            for k, al in enumerate(alphas):
                state["a"] = float(al)
                out[k, idx] = d_from_hidden(model, final_hidden(model, x), Y, N).cpu().numpy()
    finally:
        hk.remove()
    return out


def _o_weight32(model, l):
    lin = _layers(model)[l].self_attn.o_proj
    W = lin.weight
    if getattr(W, "quant_state", None) is not None:
        import bitsandbytes.functional as bnbF
        return bnbF.dequantize_4bit(W.data, W.quant_state).float()
    return W.data.float()


@torch.no_grad()
def component_writes(model, tok, id_lists, direction, up_to_layer, batch=32):
    """Projection of each head's and each MLP's output at the last token onto `direction` (unit vector),
    for the layers below `up_to_layer` (the components that write before the direction is read).

    Returns heads [n, up_to_layer, H] and mlp [n, up_to_layer]."""
    info = model_info(model)
    H, hd = info["n_heads"], info["head_dim"]
    u = torch.as_tensor(np.asarray(direction) / np.linalg.norm(direction), dtype=torch.float32, device=model.device)
    layers = _layers(model)[:up_to_layer]
    # per head: (z_h @ W_h^T) . u = z_h . (W_h^T u): precompute W_h^T u for every layer
    Wu = [(_o_weight32(model, l).view(-1, H, hd) * u[:, None, None]).sum(0) for l in range(up_to_layer)]   # [H, hd]
    n = len(id_lists)
    heads = np.zeros((n, up_to_layer, H), dtype=np.float32)
    mlp = np.zeros((n, up_to_layer), dtype=np.float32)
    store = {}
    hooks = []
    for l, layer in enumerate(layers):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
            lambda m, a, l=l: store.__setitem__(("z", l), a[0][:, -1].float())))
        hooks.append(layer.mlp.register_forward_hook(
            lambda m, a, o, l=l: store.__setitem__(("m", l), o[:, -1].float())))
    try:
        for idx in _by_length(id_lists, batch):
            x = torch.tensor([id_lists[i] for i in idx], device=model.device)
            final_hidden(model, x)
            for l in range(up_to_layer):
                z = store[("z", l)].view(len(idx), H, hd)
                heads[idx, l] = (z * Wu[l][None]).sum(-1).cpu().numpy()
                mlp[idx, l] = (store[("m", l)] @ u).cpu().numpy()
    finally:
        for hk in hooks:
            hk.remove()
    return heads, mlp


def logreg_predict(F_train, y_train, F, C=0.05, iters=200):
    """L2 logistic regression (same objective as sklearn's LogisticRegression(C=C)), fitted with L-BFGS on the GPU
    if there is one. F_train [n, k] and F [m, k] already standardised. Returns boolean predictions for F."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.as_tensor(F_train, dtype=torch.float32, device=dev)
    y = torch.as_tensor(np.asarray(y_train, dtype=np.float32), device=dev)
    w = torch.zeros(A.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, y, reduction="sum") + (w @ w) / (2 * C)
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        out = (torch.as_tensor(F, dtype=torch.float32, device=dev) @ w + b) > 0
    return out.cpu().numpy()
