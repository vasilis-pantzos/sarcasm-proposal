"""The score d = log P(Yes|yes) - log P(No|no) at the last position."""
import numpy as np
import torch

from .prompts import chat_text


def yn_ids(tok):
    """Token ids of the single-token answers Yes/yes and No/no."""
    pick = lambda ws: [t[0] for t in (tok.encode(w, add_special_tokens=False) for w in ws) if len(t) == 1]
    Y, N = pick(["Yes", "yes"]), pick(["No", "no"])
    assert Y and N, "Yes/No are not single tokens for this tokenizer"
    return Y, N


def d_from_logits(logits, Y, N):
    """logits: [batch, vocab] at the last position -> d for each row (float32, on CPU)."""
    lg = logits.float()
    return (torch.logsumexp(lg[:, Y], -1) - torch.logsumexp(lg[:, N], -1)).cpu()


def last_logits(model, input_ids, **kw):
    """Run the model and return (output, logits of the last position). Uses logits_to_keep when available."""
    try:
        out = model(input_ids, logits_to_keep=1, **kw)
    except TypeError:
        out = model(input_ids, **kw)
    return out, out.logits[:, -1, :]


@torch.no_grad()
def d_single(model, tok, ids):
    """d for one list of token ids."""
    Y, N = yn_ids(tok)
    x = torch.tensor([ids], device=model.device)
    return d_from_logits(last_logits(model, x)[1], Y, N)[0].item()


@torch.no_grad()
def d_batch(model, tok, id_lists, batch_size=16):
    """d for many token-id lists, batched with left padding (same result as d_single)."""
    Y, N = yn_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order = np.argsort([len(x) for x in id_lists])
    out = np.zeros(len(id_lists))
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        L = max(len(id_lists[j]) for j in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for b, j in enumerate(idx):
            seq = id_lists[j]
            ids[b, L - len(seq):] = torch.tensor(seq)
            mask[b, L - len(seq):] = 1
        ids, mask = ids.to(model.device), mask.to(model.device)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        _, lg = last_logits(model, ids, attention_mask=mask, position_ids=pos)
        out[idx] = d_from_logits(lg, Y, N).numpy()
    return out


def d_prompts(model, tok, prompts, batch_size=16):
    """d for raw prompt strings (applies the chat template)."""
    ids = [tok(chat_text(tok, p), add_special_tokens=False)["input_ids"] for p in prompts]
    return d_batch(model, tok, ids, batch_size)


def triple_accuracy(d_P, d_S, d_N):
    """Fraction of triples with d(S) > d(P) and d(S) > d(N)."""
    d_P, d_S, d_N = map(np.asarray, (d_P, d_S, d_N))
    return float(((d_S > d_P) & (d_S > d_N)).mean())
