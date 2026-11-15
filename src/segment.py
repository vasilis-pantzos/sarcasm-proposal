"""Phase 6a: separating a sentence into a SITUATION and a PHRASE (the evaluative words).

"Love waiting 2 hours for the bus"  ->  phrase: "Love", situation: "waiting 2 hours for the bus".

Three segmenters, all with the same output: {"situation": str or None, "phrase": str or None,
"spans": {"SIT": [(start, end), ...], "PHR": [(start, end), ...]}} (character ranges in the text).
    a) LexiconRule      transparent rule: sentiment cues (VADER lexicon + a short list of multi-word cues) and
                        punctuation. (A dependency parse was too unreliable on tweets.)
    b) QASegmenter      an off-the-shelf extractive QA model, no training: "What happened?" /
                        "How does the writer feel about it?"
    c) TokenSegmenter   a small token classifier (DistilBERT) trained on synthetic sentences with known spans.
A segmenter returns None for a part it cannot find; the mechanism then answers "not sarcastic".

Synthetic data: the 140 situations of dataset v2 placed in ~30 sentence templates (phrase before, after, or on
both sides of the situation, with and without punctuation, plus sentences with only one of the two parts). Templates,
situations and phrases are split so that the test set has templates, situations and phrases never seen in training.
"""
import random
import re

import numpy as np
import pandas as pd

LABELS = ["O", "SIT", "PHR"]
_WORD = re.compile(r"https?://\S+|[@#]?\w+(?:['’]\w+)*|[^\w\s]")


# ------------------------------------------------------------------ words and spans
def words(text):
    """[(word, start, end)]: words, hashtags, mentions, URLs, and single punctuation / emoji characters."""
    return [(m.group(), m.start(), m.end()) for m in _WORD.finditer(text)]


def labels_from_spans(text, spans):
    """Word labels ("O", "SIT", "PHR") from character spans {"SIT": [(s, e)], "PHR": [(s, e)]}."""
    out = []
    for _, s, e in words(text):
        lab = "O"
        for name in ("SIT", "PHR"):
            if any(s < b and e > a for a, b in spans.get(name, [])):
                lab = name
        out.append(lab)
    return out


def spans_from_labels(text, labels):
    """Character spans of the runs of SIT / PHR words."""
    ws = words(text)
    spans = {"SIT": [], "PHR": []}
    i = 0
    while i < len(ws):
        lab = labels[i]
        if lab in spans:
            j = i
            while j + 1 < len(ws) and labels[j + 1] == lab:
                j += 1
            spans[lab].append((ws[i][1], ws[j][2]))
            i = j + 1
        else:
            i += 1
    return spans


def smooth(labels, gap=1, max_phrase_runs=2):
    """Clean word labels: fill gaps of `gap` O-words inside a run, keep the longest situation run and the
    `max_phrase_runs` longest phrase runs."""
    lab = list(labels)
    for i in range(1, len(lab) - gap):
        if lab[i] == "O" and lab[i - 1] in ("SIT", "PHR") and lab[i - 1] == lab[min(i + gap, len(lab) - 1)]:
            lab[i] = lab[i - 1]
    runs = {"SIT": [], "PHR": []}
    i = 0
    while i < len(lab):
        if lab[i] in runs:
            j = i
            while j + 1 < len(lab) and lab[j + 1] == lab[i]:
                j += 1
            runs[lab[i]].append((j - i + 1, i, j))
            i = j + 1
        else:
            i += 1
    keep = {"SIT": sorted(runs["SIT"], reverse=True)[:1], "PHR": sorted(runs["PHR"], reverse=True)[:max_phrase_runs]}
    out = ["O"] * len(lab)
    for name, rs in keep.items():
        for _, a, b in rs:
            out[a:b + 1] = [name] * (b - a + 1)
    return out


def result(text, spans, min_words=1):
    """Common output format. The text of a part joins its runs with ' … '."""
    out = {"spans": spans}
    for name, key in (("SIT", "situation"), ("PHR", "phrase")):
        parts = [text[a:b].strip() for a, b in spans.get(name, [])]
        parts = [p for p in parts if len(words(p)) >= min_words]
        out[key] = " … ".join(parts) if parts else None
    return out


# ------------------------------------------------------------------ synthetic data
PRE_POS = ["Love", "I love", "I just love", "Gotta love", "Nothing like", "Nothing better than", "Thanks for",
           "Thank you for", "So glad about", "Can't wait for", "Nothing beats", "I really enjoy", "Yay for",
           "Living for", "Absolutely love", "Don't you just love", "Really enjoying", "So happy about"]
PRE_NEG = ["Hate", "I hate", "Can't stand", "So tired of", "Sick of", "Not a fan of", "I really dislike",
           "Ugh,", "Dreading", "So done with"]
POST_POS = ["just great", "love it", "best day ever", "so much fun", "#blessed", "living the dream", "perfect",
            "what a treat", "amazing", "just what I needed", "can't complain", "my favourite"]
POST_NEG = ["ugh", "worst day ever", "so annoying", "#fail", "awful", "this sucks", "kill me now", "not again",
            "so frustrating", "terrible"]

# a template is a list of (label, piece); pieces: {P} pre-phrase, {q} short post-phrase, {Q} full post-phrase
# sentence (from dataset v2), {S} situation, anything else is literal text (label O)
TEMPLATES = {
    "pre": [[("PHR", "{P}"), ("O", " "), ("SIT", "{S}")],
            [("PHR", "{P}"), ("O", " "), ("SIT", "{S}"), ("O", "!")],
            [("PHR", "{P}"), ("O", " "), ("SIT", "{S}"), ("O", " lol")],
            [("PHR", "{P}"), ("O", " "), ("SIT", "{S}"), ("O", " 🙃")],
            [("O", "Oh "), ("PHR", "{p}"), ("O", " "), ("SIT", "{S}"), ("O", ".")],
            [("O", "Honestly, "), ("PHR", "{p}"), ("O", " "), ("SIT", "{S}"), ("O", ".")],
            [("PHR", "{P}"), ("O", " "), ("SIT", "{S}"), ("O", " today")],
            [("O", "@friend "), ("PHR", "{p}"), ("O", " "), ("SIT", "{S}"), ("O", " haha")]],
    "post": [[("SIT", "{Sc}"), ("O", ". "), ("PHR", "{Q}")],
             [("SIT", "{Sc}"), ("O", ", "), ("PHR", "{q}"), ("O", ".")],
             [("SIT", "{Sc}"), ("O", "... "), ("PHR", "{q}")],
             [("SIT", "{Sc}"), ("O", "! "), ("PHR", "{qc}"), ("O", "!")],
             [("SIT", "{S}"), ("O", " - "), ("PHR", "{q}")],
             [("O", "Today was all about "), ("SIT", "{S}"), ("O", ". "), ("PHR", "{Q}")],
             [("O", "Guess what my evening looked like: "), ("SIT", "{S}"), ("O", ". "), ("PHR", "{Q}")],
             [("O", "Me today: "), ("SIT", "{S}"), ("O", ". "), ("PHR", "{qc}")],
             [("SIT", "{Sc}"), ("O", " "), ("PHR", "{q}"), ("O", " 😂")],
             [("SIT", "{Sc}"), ("O", " this morning. "), ("PHR", "{Q}")]],
    "both": [[("PHR", "{P}"), ("O", " "), ("SIT", "{S}"), ("O", ", "), ("PHR", "{q}")],
             [("PHR", "{P}"), ("O", " "), ("SIT", "{S}"), ("O", ". "), ("PHR", "{Q}")],
             [("PHR", "{qc}"), ("O", ", "), ("PHR", "{p}"), ("O", " "), ("SIT", "{S}"), ("O", ".")]],
    "only_sit": [[("O", "Spent the afternoon "), ("SIT", "{S}"), ("O", ".")],
                 [("SIT", "{Sc}"), ("O", " today.")],
                 [("O", "Just "), ("SIT", "{S}"), ("O", "...")]],
    "only_phr": [[("PHR", "{Q}")],
                 [("PHR", "{qc}"), ("O", "!")]],
}


def _cap(s):
    return s[:1].upper() + s[1:]


def make_synthetic(sit_df, phr_full, n_per_template=60, test_frac=0.3, seed=0):
    """Synthetic sentences with known spans.
    sit_df: situations (columns situation, situation_type); phr_full: full phrase sentences (columns frame,
    frame_type), e.g. the phrases of dataset v2. Returns a DataFrame with text, labels, family, template, split."""
    rng = random.Random(seed)
    def split(items):
        items = list(items); rng.shuffle(items)
        k = max(1, int(round(len(items) * test_frac)))
        return {"test": items[:k], "train": items[k:]}
    sits = split(sit_df[["situation", "situation_type"]].drop_duplicates().itertuples(index=False))
    full = split(phr_full[["frame", "frame_type"]].drop_duplicates().itertuples(index=False))
    pre = split([(p, "positive") for p in PRE_POS] + [(p, "negative") for p in PRE_NEG])
    post = split([(p, "positive") for p in POST_POS] + [(p, "negative") for p in POST_NEG])
    tmpl = {fam: split(range(len(ts))) for fam, ts in TEMPLATES.items()}
    rows = []
    for part in ("train", "test"):
        for fam, ts in TEMPLATES.items():
            for t_id in tmpl[fam][part]:
                for _ in range(n_per_template if part == "train" else max(10, n_per_template // 3)):
                    s, s_type = rng.choice(sits[part])
                    P, p_type = rng.choice(pre[part])
                    q, q_type = rng.choice(post[part])
                    Q, Q_type = rng.choice(full[part])
                    fill = {"{P}": P, "{p}": P[:1].lower() + P[1:] if not P.startswith("I ") else P,
                            "{q}": q, "{qc}": _cap(q), "{Q}": Q, "{S}": s, "{Sc}": _cap(s)}
                    text, spans = "", {"SIT": [], "PHR": []}
                    for lab, piece in ts[t_id]:
                        val = fill.get(piece, piece)
                        if lab in spans:
                            spans[lab].append((len(text), len(text) + len(val)))
                        text += val
                    rows.append(dict(text=text, labels=labels_from_spans(text, spans), family=fam,
                                     template=f"{fam}{t_id}", split=part, situation=s, situation_type=s_type,
                                     has_both=fam in ("pre", "post", "both")))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ evaluation
def evaluate(texts, gold_labels, preds):
    """Word-level precision / recall / F1 for SIT and PHR, and how often 'both parts found' matches the gold."""
    out = {}
    pl = [labels_from_spans(t, p["spans"]) for t, p in zip(texts, preds)]
    for lab in ("SIT", "PHR"):
        tp = sum(sum(g == lab and q == lab for g, q in zip(G, P)) for G, P in zip(gold_labels, pl))
        fp = sum(sum(g != lab and q == lab for g, q in zip(G, P)) for G, P in zip(gold_labels, pl))
        fn = sum(sum(g == lab and q != lab for g, q in zip(G, P)) for G, P in zip(gold_labels, pl))
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[f"{lab} precision"], out[f"{lab} recall"] = pr, rc
        out[f"{lab} F1"] = 2 * pr * rc / max(pr + rc, 1e-9)
    g_both = np.array([("SIT" in G) and ("PHR" in G) for G in gold_labels])
    p_both = np.array([p["situation"] is not None and p["phrase"] is not None for p in preds])
    out["both found: accuracy"] = float((g_both == p_both).mean())
    out["both found: recall"] = float(p_both[g_both].mean()) if g_both.any() else np.nan
    out["both found: false alarms"] = float(p_both[~g_both].mean()) if (~g_both).any() else np.nan
    return out


# ------------------------------------------------------------------ a) transparent rule
POS_CUES = ["nothing like", "nothing better than", "nothing beats", "gotta love", "can't wait", "cant wait",
            "just what i needed", "living the dream", "made my day", "so glad", "so happy", "thanks for",
            "thank you for", "yay", "blessed", "love", "loving", "loved", "best", "great", "perfect", "amazing",
            "wonderful", "fantastic", "awesome", "brilliant", "fun", "favourite", "favorite", "enjoy", "enjoying"]
NEG_CUES = ["ugh", "hate", "can't stand", "cant stand", "sick of", "tired of", "worst", "awful", "terrible", "sucks"]
MODIFIERS = {"i", "i'm", "im", "just", "so", "really", "oh", "wow", "absolutely", "totally", "don't", "you", "it",
             "am", "is", "was", "such", "a", "an", "the", "what", "how", "isn't", "my", "we", "so", "very", "!"}
LINKERS = {"for", "like", "than", "about", "how", "when", "that", "it", "to", "of", "with"}
_END = {".", "!", "?", "…"}


class LexiconRule:
    """Find the strongest sentiment cue; the phrase is the cue with its modifiers (or its whole short sentence),
    the situation is what follows it in the same sentence, else the longest other sentence."""

    def __init__(self, min_valence=1.5):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.lex = SentimentIntensityAnalyzer().lexicon
        self.min_valence = min_valence

    def _cues(self, ws):
        low = [w.lower().lstrip("#").replace("’", "'") for w, _, _ in ws]
        found = []
        for cues, sign in ((POS_CUES, 1), (NEG_CUES, -1)):
            for c in cues:
                toks = c.split()
                for i in range(len(low) - len(toks) + 1):
                    if low[i:i + len(toks)] == toks:
                        found.append((2.5, i, i + len(toks), sign))
        for i, w in enumerate(low):
            v = self.lex.get(w)
            if v is not None and abs(v) >= self.min_valence:
                found.append((abs(v), i, i + 1, 1 if v > 0 else -1))
        return found

    def __call__(self, text):
        ws = words(text)
        empty = {"SIT": [], "PHR": []}
        if not ws:
            return result(text, empty)
        # sentences (hard boundaries)
        sent = np.zeros(len(ws), dtype=int)
        k = 0
        for i, (w, _, _) in enumerate(ws):
            sent[i] = k
            if w in _END and i + 1 < len(ws) and ws[i + 1][0] not in _END:
                k += 1
        cues = self._cues(ws)
        if not cues:
            return result(text, empty)
        _, a, b, _ = max(cues, key=lambda c: (c[0], c[2] - c[1]))
        s_id = sent[a]
        idx = [i for i in range(len(ws)) if sent[i] == s_id]
        lo, hi = a, b                               # phrase = [lo, hi)
        while lo - 1 >= idx[0] and ws[lo - 1][0].lower() in MODIFIERS:
            lo -= 1
        while hi < idx[-1] + 1 and ws[hi][0].lower() in LINKERS:
            hi += 1
        content = lambda r: [i for i in r if re.match(r"\w", ws[i][0]) and not ws[i][0].startswith(("@", "http"))]
        after = content(range(hi, idx[-1] + 1))
        before = content(range(idx[0], lo))
        sit = None
        if len(after) >= 2:
            # stop at a comma followed by another cue (e.g. "Love X, just great")
            stop = idx[-1] + 1
            for i in range(hi, idx[-1] + 1):
                if ws[i][0] == "," and any(c[1] > i for c in cues if sent[c[1]] == s_id):
                    stop = i; break
            sit = [i for i in after if i < stop]
        if (sit is None or len(sit) < 2) and len(before) >= 2:
            # the phrase ends the sentence: "X, just great" -> situation before it (drop a leading frame "...:")
            colon = [i for i in range(idx[0], lo) if ws[i][0] == ":"]
            start = colon[-1] + 1 if colon else idx[0]
            sit = [i for i in before if i >= start]
        if sit is None or len(sit) < 2:
            # whole short sentence is the phrase; situation = longest other sentence
            lo, hi = idx[0], idx[-1] + 1
            others = {}
            for i in range(len(ws)):
                if sent[i] != s_id:
                    others.setdefault(sent[i], []).append(i)
            best = max(others.values(), key=lambda r: len(content(r)), default=[])
            best = content(best)
            colon = [i for i in best if i > 0 and ws[i - 1][0] == ":"]
            if colon:
                best = [i for i in best if i >= colon[0]]
            sit = best if len(best) >= 2 else None
        if sit is None:
            return result(text, {"SIT": [], "PHR": [(ws[lo][1], ws[hi - 1][2])]})
        return result(text, {"SIT": [(ws[sit[0]][1], ws[sit[-1]][2])], "PHR": [(ws[lo][1], ws[hi - 1][2])]})


# ------------------------------------------------------------------ b) extractive QA, no training
QA_MODEL = "distilbert-base-cased-distilled-squad"


class QASegmenter:
    """Extractive QA without the transformers `pipeline` (removed in transformers 5): the best answer span is the
    pair (start <= end, at most `max_len` tokens) with the highest start + end logit inside the context; its score
    is p(start) * p(end)."""

    def __init__(self, model=QA_MODEL, q_sit="What happened?", q_phr="How does the writer feel about it?",
                 min_score=0.05, device=None, max_len=30):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model).to(self.device).eval()
        self.q_sit, self.q_phr, self.min_score, self.max_len = q_sit, q_phr, min_score, max_len

    def _answers(self, question, contexts):
        torch = self.torch
        enc = self.tok([question] * len(contexts), list(contexts), truncation="only_second", max_length=384,
                       padding=True, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping").numpy()
        seq = [enc.sequence_ids(b) for b in range(len(contexts))]
        if self.model.config.model_type in ("distilbert", "roberta"):
            enc.pop("token_type_ids", None)
        with torch.no_grad():
            o = self.model(**{k: v.to(self.device) for k, v in enc.items()})
        S, E = o.start_logits.float().cpu().numpy(), o.end_logits.float().cpu().numpy()
        out = []
        for b in range(len(contexts)):
            ctx = np.array([x == 1 for x in seq[b]])
            if not ctx.any():
                out.append({"score": 0.0, "start": 0, "end": 0}); continue
            s, e = np.where(ctx, S[b], -1e9), np.where(ctx, E[b], -1e9)
            ps = np.exp(s - s.max()); ps /= ps.sum()
            pe = np.exp(e - e.max()); pe /= pe.sum()
            T = len(s)
            M = s[:, None] + e[None, :]
            ii, jj = np.indices((T, T))
            M[(jj < ii) | (jj - ii >= self.max_len)] = -np.inf
            i, j = np.unravel_index(np.argmax(M), M.shape)
            out.append({"score": float(ps[i] * pe[j]), "start": int(offs[b, i, 0]), "end": int(offs[b, j, 1])})
        return out

    def batch(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            part = list(texts[i:i + bs])
            a_s = self._answers(self.q_sit, part)
            a_p = self._answers(self.q_phr, part)
            for t, s, p in zip(part, a_s, a_p):
                spans = {"SIT": [], "PHR": []}
                if p["score"] >= self.min_score and p["end"] > p["start"]:
                    spans["PHR"].append((p["start"], p["end"]))
                if s["score"] >= self.min_score:
                    a, b = s["start"], s["end"]
                    for pa, pb in spans["PHR"]:            # remove the overlap with the phrase
                        if a < pb and b > pa:
                            a, b = (pb, b) if b - pb >= pa - a else (a, pa)
                    if b > a:
                        spans["SIT"].append((a, b))
                r = result(t, spans)
                r["scores"] = (s["score"], p["score"])
                out.append(r)
        return out

    def fit_threshold(self, texts, has_both, grid=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)):
        """Choose min_score on training sentences: the value whose 'both found' decision agrees most often with
        has_both (whether the sentence really has both parts)."""
        keep = self.min_score
        self.min_score = 0.0
        sc = np.array([r["scores"] for r in self.batch(list(texts))])
        has_both = np.asarray(has_both, bool)
        acc = {g: float(((sc.min(1) >= g) == has_both).mean()) for g in grid}
        self.min_score = max(acc, key=acc.get) if acc else keep
        return acc

    def __call__(self, text):
        return self.batch([text])[0]


# ------------------------------------------------------------------ c) token classifier
TOKEN_MODEL = "distilbert-base-uncased"


class TokenSegmenter:
    """A token classifier (word labels O / SIT / PHR) trained on the synthetic sentences."""

    def __init__(self, model=TOKEN_MODEL, device=None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
            label2id={l: i for i, l in enumerate(LABELS)}).to(self.device)

    def _encode(self, texts, labels=None):
        ws = [[w for w, _, _ in words(t)] for t in texts]
        enc = self.tok(ws, is_split_into_words=True, truncation=True, max_length=128, padding=True,
                       return_tensors="pt")
        if self.model.config.model_type in ("distilbert", "roberta"):
            enc.pop("token_type_ids", None)
        if labels is not None:
            y = np.full(enc["input_ids"].shape, -100)
            for b in range(len(texts)):
                prev = None
                for j, wid in enumerate(enc.word_ids(b)):
                    if wid is not None and wid != prev:
                        y[b, j] = LABELS.index(labels[b][wid])
                    prev = wid
            enc["labels"] = self.torch.as_tensor(y)
        return enc

    def train(self, texts, labels, epochs=3, lr=5e-5, bs=32, seed=0, progress=True):
        torch = self.torch
        torch.manual_seed(seed)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        n = len(texts)
        steps = epochs * ((n + bs - 1) // bs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
        rng = np.random.default_rng(seed)
        self.model.train()
        for ep in range(epochs):
            order = rng.permutation(n)
            tot = 0.0
            for i in range(0, n, bs):
                idx = order[i:i + bs]
                enc = self._encode([texts[j] for j in idx], [labels[j] for j in idx]).to(self.device)
                loss = self.model(**enc).loss
                loss.backward(); opt.step(); sched.step(); opt.zero_grad()
                tot += float(loss) * len(idx)
            if progress:
                print(f"  epoch {ep + 1}: loss {tot / n:.4f}")
        self.model.eval()

    def batch(self, texts, bs=64):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                part = list(texts[i:i + bs])
                enc = self._encode(part).to(self.device)
                pred = self.model(**enc).logits.argmax(-1).cpu().numpy()
                for b, t in enumerate(part):
                    n_w = len(words(t))
                    lab = ["O"] * n_w
                    prev = None
                    for j, wid in enumerate(enc.word_ids(b)):
                        if wid is not None and wid != prev and wid < n_w:
                            lab[wid] = LABELS[pred[b, j]]
                        prev = wid
                    out.append(result(t, spans_from_labels(t, smooth(lab))))
        return out

    def __call__(self, text):
        return self.batch([text])[0]

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


# ------------------------------------------------------------------ iSarcasmEval
ISARCASM_REPO = "https://github.com/iabufarha/iSarcasmEval.git"
CATEGORIES = ["sarcasm", "irony", "satire", "understatement", "overstatement", "rhetorical_question"]


def load_isarcasm(root):
    """English test set: task A (text, sarcastic) with the task B categories, and the task C pairs."""
    import os
    a = pd.read_csv(os.path.join(root, "test", "task_A_En_test.csv"))
    b = pd.read_csv(os.path.join(root, "test", "task_B_En_test.csv"))
    b = b.groupby("text", as_index=False)[CATEGORIES].max()          # the B file is in another order
    test = a.merge(b, on="text", how="left")
    test[CATEGORIES] = test[CATEGORIES].fillna(0).astype(int)
    assert len(test) == len(a)
    pairs = pd.read_csv(os.path.join(root, "test", "task_C_En_test.csv"))
    train = pd.read_csv(os.path.join(root, "train", "train.En.csv")).drop(columns=["Unnamed: 0"], errors="ignore")
    return test, pairs, train
