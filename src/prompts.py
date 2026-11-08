"""From the dataset to token ids, regions and clean/corrupted pairs."""
import os
from dataclasses import dataclass, field

import pandas as pd

REGIONS = ["situation", "phrase", "question", "final"]
QUESTION_END = "Answer with Yes or No."
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sarcasm_dataset.csv")

# experiment name -> (condition of the corrupted sentence, region that differs, token-matched column)
EXPERIMENTS = {
    "situation": ("P", "situation", "token_matched_PS"),   # S vs P: only the situation changes
    "phrase": ("N", "phrase", "token_matched_SN"),         # S vs N: only the phrase changes
}


def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    need = {"triple_id", "condition", "text", "prompt", "situation", "frame", "split",
            "token_matched_PS", "token_matched_SN"}
    missing = need - set(df.columns)
    assert not missing, f"missing columns: {missing}"
    return df


def chat_text(tok, prompt):
    """The prompt inside the model's chat template, ready for the answer token."""
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@dataclass
class Encoded:
    ids: list                       # token ids of the full chat prompt
    regions: dict                   # region name -> list of token positions
    text: str = ""                  # the sentence (for printing)


def encode(tok, row):
    """One dataset row -> token ids + the positions of each region.

    Regions: the situation words, the phrase words, the question that follows
    (without the tokens shared with the phrase), and the final token.
    """
    text = chat_text(tok, row.prompt)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    u0 = text.index(row.text)
    s0 = u0 + row.text.index(row.situation); s1 = s0 + len(row.situation)
    f0 = u0 + row.text.index(row.frame);     f1 = f0 + len(row.frame)
    q1 = text.index(QUESTION_END) + len(QUESTION_END)
    toks = lambda a, b: [i for i, (x, y) in enumerate(offs) if y > a and x < b]
    phrase = toks(f0, f1)
    regions = {"situation": toks(s0, s1), "phrase": phrase,
               "question": [i for i in toks(f1, q1) if i not in phrase],
               "final": [len(ids) - 1]}
    return Encoded(list(ids), regions, row.text)


def aligned(a, b, allowed):
    """Same length, and the two id lists differ only at the allowed positions."""
    allowed = set(allowed)
    return len(a) == len(b) and all(i in allowed for i, (x, y) in enumerate(zip(a, b)) if x != y)


@dataclass
class Pair:
    experiment: str                 # "situation" or "phrase"
    triple_id: int
    clean: Encoded                  # the sarcastic sentence S
    corrupted: Encoded              # P or N
    split: str = "discovery"
    info: dict = field(default_factory=dict)   # filled later, e.g. d_clean, d_corrupted


def make_pairs(df, tok, experiment, split="discovery", token_matched=True):
    """Clean (S) / corrupted (P or N) pairs for one experiment.

    token_matched=True keeps only pairs with the same number of tokens that differ only
    in the changed region (needed for position-by-position patching).
    """
    corr_cond, region, col = EXPERIMENTS[experiment]
    sub = df if split is None else df[df.split == split]
    rows = {(r.triple_id, r.condition): r for r in sub.itertuples()}
    pairs = []
    for tid in sorted(sub.triple_id.unique()):
        s, c = rows[(tid, "S")], rows[(tid, corr_cond)]
        if token_matched and not bool(getattr(s, col)):
            continue
        es, ec = encode(tok, s), encode(tok, c)
        if token_matched and not aligned(es.ids, ec.ids, es.regions[region]):
            continue
        pairs.append(Pair(experiment, int(tid), es, ec, s.split))
    return pairs


def describe(tok, enc):
    """Readable view of the regions of one encoded prompt."""
    return {r: [tok.decode([enc.ids[p]]) for p in enc.regions[r]] for r in REGIONS}
