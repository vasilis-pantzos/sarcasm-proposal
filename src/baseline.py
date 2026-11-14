"""Phase 3d: an interpretable-by-design baseline.

Two small sentiment classifiers, one for the situation and one for the phrase, and a fixed rule on top:
    "disagree"    : sarcastic if the two sentiments disagree (positive phrase + unpleasant situation, or the reverse)
    "directional" : sarcastic only if the phrase is positive and the situation unpleasant
The two rules differ only on Q (pleasant situation + negative phrase).

The classifiers get the situation and the phrase ALREADY separated (taken from the dataset).

Classifiers; each gives p = probability of "pleasant" (situation) / "positive" (phrase), label = p > 0.5:
    A  VADER   : lexicon + rules, no training. p = (compound + 1) / 2, so a text with no sentiment word (compound 0)
                 counts as unpleasant / negative.
    B  SST-2   : DistilBERT fine-tuned on SST-2 (movie reviews), no training on our data.
    C  MiniLM  : all-MiniLM-L6-v2 sentence embeddings + logistic regression trained on our labels. Every item of the
                 dataset is predicted by a model that did NOT see it (k-fold over the training items).
"""
import numpy as np

SST2 = "distilbert-base-uncased-finetuned-sst-2-english"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
_CACHE = {}


def _device(device=None):
    import torch
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def vader_p(texts):
    if "vader" not in _CACHE:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _CACHE["vader"] = SentimentIntensityAnalyzer()
    a = _CACHE["vader"]
    return np.array([(a.polarity_scores(t)["compound"] + 1) / 2 for t in texts])


def vader_words(text):
    """The words of `text` that are in the VADER lexicon, with their valence (for explaining a decision)."""
    vader_p([""])
    lex = _CACHE["vader"].lexicon
    return {w: lex[w] for w in (x.strip(".,!?;:\"'").lower() for x in text.split()) if w in lex}


def sst2_p(texts, device=None, batch=64):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if "sst2" not in _CACHE:
        tok = AutoTokenizer.from_pretrained(SST2)
        m = AutoModelForSequenceClassification.from_pretrained(SST2).to(_device(device)).eval()
        _CACHE["sst2"] = (tok, m)
    tok, m = _CACHE["sst2"]
    pos = m.config.label2id.get("POSITIVE", 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            x = tok(list(texts[i:i + batch]), return_tensors="pt", padding=True, truncation=True).to(m.device)
            out.append(m(**x).logits.float().softmax(-1)[:, pos].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def minilm_embed(texts, device=None):
    if "minilm" not in _CACHE:
        from sentence_transformers import SentenceTransformer
        _CACHE["minilm"] = SentenceTransformer(MINILM, device=_device(device))
    return np.asarray(_CACHE["minilm"].encode(list(texts), normalize_embeddings=True, batch_size=64))


def fit_logreg(E, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=2000).fit(E, np.asarray(y).astype(int))


def oof_p(E, y, train, folds=5, C=1.0, seed=0):
    """Out-of-sample p for every item. Training items (train=True) get a stratified k-fold prediction; the other
    items get the prediction of a model trained on all training items."""
    E, y, train = np.asarray(E), np.asarray(y).astype(bool), np.asarray(train).astype(bool)
    p = np.zeros(len(E))
    tr = np.where(train)[0]
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(tr), dtype=int)
    for lab in (False, True):
        k = np.where(y[tr] == lab)[0]
        fold[rng.permutation(k)] = np.arange(len(k)) % folds
    for f in range(folds):
        test = tr[fold == f]
        if len(test):
            p[test] = fit_logreg(E[tr[fold != f]], y[tr[fold != f]], C).predict_proba(E[test])[:, 1]
    if (~train).any():
        p[~train] = fit_logreg(E[tr], y[tr], C).predict_proba(E[~train])[:, 1]
    return p


def apply_rules(p_sit, p_phr):
    """Decisions and continuous scores of the two rules (p = probability pleasant / positive)."""
    p_sit, p_phr = np.asarray(p_sit, float), np.asarray(p_phr, float)
    pl, po = p_sit > 0.5, p_phr > 0.5
    return {
        "disagree": po != pl,
        "directional": po & ~pl,
        "disagree score": p_phr * (1 - p_sit) + (1 - p_phr) * p_sit,
        "directional score": p_phr * (1 - p_sit),
    }


def cohen_kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def compare(pred, ref, truth):
    """pred: decisions of a system; ref: decisions of the reference system; truth: ground truth.
    Agreement, kappa, and whether the system fails on the same prompts as the reference."""
    pred, ref, truth = (np.asarray(x, bool) for x in (pred, ref, truth))
    err_p, err_r = pred != truth, ref != truth
    both = int((err_p & err_r).sum())
    exp = err_p.mean() * err_r.sum()          # shared errors expected if the errors were independent
    return {
        "agreement": float((pred == ref).mean()),
        "kappa": cohen_kappa(pred, ref),
        "errors (system)": int(err_p.sum()),
        "errors (reference)": int(err_r.sum()),
        "shared errors": both,
        "shared if independent": float(exp),
        "P(system wrong | ref wrong)": float(err_p[err_r].mean()) if err_r.any() else np.nan,
        "P(system wrong)": float(err_p.mean()),
    }


def triple_acc(yes, frame):
    """yes: decisions for the rows of `frame` (columns triple_id, condition in P/S/N). Share of triples with
    P = No, S = Yes, N = No."""
    t = frame.assign(yes=np.asarray(yes, bool)).pivot_table(index="triple_id", columns="condition", values="yes")
    t = t.dropna(subset=["P", "S", "N"])
    return float(((t.P == 0) & (t.S == 1) & (t.N == 0)).mean())
