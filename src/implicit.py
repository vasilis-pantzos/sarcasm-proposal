"""Phase 5: the implicit question.

Instead of "Is this person being sarcastic?" the model is asked a question whose correct answer needs the sarcasm
to be understood, e.g. "Is this person actually happy about it?". The correct answer follows the SITUATION, not the
words of the phrase:
    P (pleasant + positive)   -> Yes
    S (unpleasant + positive) -> No      <- a literal reading of the phrase says Yes
    N (unpleasant + negative) -> No
    Q (pleasant + negative)   -> ambiguous (no ground truth)
"""
import numpy as np
import torch

from .direction import _by_length
from .eap import _layers
from .metric import d_from_hidden, final_hidden, yn_ids

SARCASM_Q = "Is this person being sarcastic?"
IMPLICIT_Q = "Is this person actually happy about it?"


def implicit_prompt(prompt, question=IMPLICIT_Q):
    """The same prompt with the sarcasm question replaced by `question`."""
    assert SARCASM_Q in prompt, "the prompt does not contain the sarcasm question"
    return prompt.replace(SARCASM_Q, question)


@torch.no_grad()
def project_out_d(model, tok, id_lists, layers, direction, batch=32):
    """d of every prompt when the component along `direction` is removed from the residual stream at the last
    token, at the input of each layer in `layers`."""
    Y, N = yn_ids(tok)
    u = np.asarray(direction, dtype=np.float32)
    u = torch.as_tensor(u / np.linalg.norm(u), device=model.device)

    def pre(m, a):
        h = a[0].clone()
        x = h[:, -1].float()
        h[:, -1] = (x - (x @ u)[:, None] * u[None]).to(h.dtype)
        return (h,) + tuple(a[1:])

    hs = [_layers(model)[l].register_forward_pre_hook(pre) for l in layers]
    out = np.zeros(len(id_lists), dtype=np.float32)
    try:
        for idx in _by_length(id_lists, batch):
            x = torch.tensor([id_lists[i] for i in idx], device=model.device)
            out[idx] = d_from_hidden(model, final_hidden(model, x), Y, N).cpu().numpy()
    finally:
        for h in hs:
            h.remove()
    return out


def truth_implicit(frame):
    """Correct answer to the implicit question (True = Yes): the situation is pleasant. NaN-free for P, S, N, Q."""
    return (frame.situation_type == "pleasant").values


def summary_implicit(yes, frame):
    """Metrics of one system on the rows of `frame` (columns condition in P/S/N/Q, triple_id, situation_type,
    frame_type). `yes`: the system's answers (True = Yes, i.e. happy)."""
    yes = np.asarray(yes, bool)
    cond = frame.condition.values
    psn = cond != "Q"
    truth = truth_implicit(frame)
    y, t = yes[psn], truth[psn]
    out = {f"acc {c}": float((yes[cond == c] == truth[cond == c]).mean()) for c in "PSN"}
    out["bal. acc"] = 0.5 * (y[t].mean() + (~y[~t]).mean())
    # triple: P Yes, S No, N No
    tab = frame[psn].assign(yes=y).pivot_table(index="triple_id", columns="condition", values="yes")
    tab = tab.dropna(subset=["P", "S", "N"])
    out["triple acc"] = float(((tab.P == 1) & (tab.S == 0) & (tab.N == 0)).mean())
    # the "No" class (not happy): S and N
    no_pred, no_true = ~y, ~t
    out["precision No"] = float((no_pred & no_true).sum() / max(no_pred.sum(), 1))
    out["recall No"] = float((no_pred & no_true).sum() / max(no_true.sum(), 1))
    # literal reading: the answer follows the phrase (only S separates the two readings)
    out["literal on S"] = float(yes[cond == "S"].mean())
    out["Yes on Q"] = float(yes[cond == "Q"].mean()) if (cond == "Q").any() else np.nan
    return out
