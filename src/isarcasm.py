"""Phase 6b: Qwen, the circuit and the custom mechanism on real tweets (iSarcasmEval).

Every tweet goes into the same prompt as dataset v2:
    Someone said: "<tweet>"
    Is this person being sarcastic? Answer with Yes or No.

Custom mechanism (no training, no threshold):
    segmenter c of 06a -> phrase and situation of the tweet
    sentiment model     -> is the phrase positive? is the situation negative?
    rule                -> sarcastic = both parts found AND positive phrase AND negative situation
(the "directional" rule of 3d: exactly "positive words about a negative situation").
"""
import numpy as np

from .implicit import SARCASM_Q, implicit_prompt
from .prompts import chat_text

PROMPT = 'Someone said: "{}"\n' + SARCASM_Q + " Answer with Yes or No."
TWITTER = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_CACHE = {}


def clean(t):
    return " ".join(str(t).split())          # one line, no double spaces


def tweet_prompt(text, question=SARCASM_Q):
    p = PROMPT.format(clean(text))
    return p if question == SARCASM_Q else implicit_prompt(p, question)


def tweet_ids(tok, texts, question=SARCASM_Q):
    """Token ids of every prompt, and the number of template tokens before / after the tweet that are the same
    for every prompt (needed to align the mean reference of the circuit across lengths)."""
    ids, pre, suf = [], [], []
    for t in texts:
        full = chat_text(tok, tweet_prompt(t, question))
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        a = full.index('Someone said: "') + len('Someone said: "')
        b = full.index('"\n', a)
        offs = enc["offset_mapping"]
        ids.append(list(enc["input_ids"]))
        pre.append(sum(1 for x, y in offs if y <= a))
        suf.append(sum(1 for x, y in offs if x >= b))
    return ids, min(pre), min(suf)


def twitter_sentiment(texts, device=None, batch=64):
    """p(negative), p(neutral), p(positive) of every text (twitter-RoBERTa, trained on tweets)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if "tw" not in _CACHE:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(TWITTER)
        m = AutoModelForSequenceClassification.from_pretrained(TWITTER).to(dev).eval()
        _CACHE["tw"] = (tok, m)
    tok, m = _CACHE["tw"]
    lab = {v.lower(): k for k, v in m.config.id2label.items()}
    cols = [lab["negative"], lab["neutral"], lab["positive"]]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            x = tok([clean(t) for t in texts[i:i + batch]], return_tensors="pt", padding=True, truncation=True,
                    max_length=128).to(m.device)
            out.append(m(**x).logits.float().softmax(-1)[:, cols].cpu().numpy())
    out = np.concatenate(out) if out else np.zeros((0, 3))
    return {"neg": out[:, 0], "neu": out[:, 1], "pos": out[:, 2]}


def mechanism(seg, sent):
    """seg: DataFrame with columns phrase, situation (None = not found). sent: {text: (neg, neu, pos)}.

    Returns a DataFrame: found (both parts), phrase positive, situation negative, sarcastic (the rule),
    score (p_pos(phrase) * p_neg(situation), 0 if a part is missing; used only to rank the two texts of a pair),
    happy (answer to the implicit question: the situation is positive, or the whole tweet if no parts)."""
    import pandas as pd
    get = lambda t, k: sent[t][k] if isinstance(t, str) else np.nan
    found = seg.phrase.notna().values & seg.situation.notna().values
    ph_pos = np.array([get(t, 2) > get(t, 0) if f else False for t, f in zip(seg.phrase, found)])
    si_neg = np.array([get(t, 0) > get(t, 2) if f else False for t, f in zip(seg.situation, found)])
    score = np.array([get(p, 2) * get(s, 0) if f else 0.0 for p, s, f in zip(seg.phrase, seg.situation, found)])
    whole_pos = np.array([sent[t][2] > sent[t][0] for t in seg.text])
    si_pos = np.array([get(s, 2) > get(s, 0) if f else False for s, f in zip(seg.situation, found)])
    return pd.DataFrame({"found": found, "phrase positive": ph_pos, "situation negative": si_neg,
                         "sarcastic": found & ph_pos & si_neg, "score": score,
                         "happy": np.where(found, si_pos, whole_pos), "literal happy": whole_pos})


def metrics(pred, y):
    """F1 / precision / recall of the sarcastic class, and accuracy."""
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    tp = (pred & y).sum()
    p = tp / max(pred.sum(), 1)
    r = tp / max(y.sum(), 1)
    return {"F1": float(2 * p * r / max(p + r, 1e-12)), "precision": float(p), "recall": float(r),
            "accuracy": float((pred == y).mean()), "said Yes": float(pred.mean())}


def best_threshold(score, y):
    """The threshold on d that gives the best F1 (sarcastic = d > threshold)."""
    s = np.sort(np.unique(score))
    cand = np.concatenate([[s[0] - 1], (s[1:] + s[:-1]) / 2]) if len(s) > 1 else s - 1
    f = [metrics(score > c, y)["F1"] for c in cand]
    return float(cand[int(np.argmax(f))])


def pick_pair(s0, s1):
    """Which text of a pair a system calls sarcastic: 0, 1, or 0.5 on a tie."""
    s0, s1 = np.asarray(s0, float), np.asarray(s1, float)
    return np.where(s1 > s0, 1.0, np.where(s0 > s1, 0.0, 0.5))
