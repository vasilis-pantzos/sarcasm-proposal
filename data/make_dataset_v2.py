"""Dataset v2: more phrases, more sentence frames, held-out phrases AND a held-out frame.

    python data/make_dataset_v2.py            (needs the Qwen2.5 tokenizer: transformers + internet)

Same design as v1 (notebooks/data.ipynb): for a pair of situations (pleasant / unpleasant) and a positive
phrase we build a triple
    P = pleasant situation + positive phrase   (not sarcastic)
    S = unpleasant situation + positive phrase (sarcastic)
    N = unpleasant situation + negative phrase (not sarcastic)
S differs from P only in the situation, and from N only in the phrase. For every positive phrase there are
several negative versions; we take the first with the same number of tokens as S (token-matched), so that
position-by-position patching works.

New in v2:
* 70 situation pairs (v1: 46), 24 positive phrases (v1: 5), 5 sentence frames (v1: 1);
* splits:  discovery        = discovery phrase + discovery frame
           heldout (phrase) = one of 6 held-out phrases (never in discovery) + a discovery frame
           heldout (frame)  = a discovery phrase + the held-out frame (never in discovery)
* per situation pair: 9 discovery triples, 3 held-out-phrase triples, 1 held-out-frame triple (seeded).
"""
import os
import random

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "sarcasm_dataset_v2.csv")
SEED = 0

PAIRS = [  # (pleasant, unpleasant) - the 46 pairs of v1 (some reworded so that both have the same tokens) ...
    ("relaxing on the beach", "waiting in the rain"), ("eating fresh pizza", "eating burnt toast"),
    ("getting a day off", "working on a Sunday"), ("winning the game", "losing my wallet"),
    ("sleeping in late", "waking up early"), ("a quiet evening at home", "a long meeting at work"),
    ("finding a parking spot", "getting a parking ticket"), ("a hot shower", "a cold shower"),
    ("getting good news", "getting more homework"), ("a fast internet connection", "a slow internet connection"),
    ("my train arriving early", "my train arriving late"), ("a sunny weekend", "a rainy weekend"),
    ("passing the exam", "failing the exam"), ("hot coffee in the morning", "cold coffee in the morning"),
    ("a clean kitchen", "a dirty kitchen"), ("friendly neighbors", "noisy neighbors"),
    ("a short queue", "a long queue"), ("getting a gift", "getting a bill"),
    ("a free upgrade", "a cancelled flight"), ("my phone working", "my phone breaking"),
    ("a delicious dinner", "a burnt dinner"), ("a relaxing massage", "a dentist appointment"),
    ("finding my lost keys", "losing my house keys"), ("a free concert", "a power outage"),
    ("my team winning", "my team losing"), ("getting a raise", "getting laid off"),
    ("a fresh haircut", "a terrible haircut"), ("a warm blanket", "a broken heater"),
    ("a peaceful walk", "a muddy walk"), ("a fast checkout", "a slow checkout"),
    ("a clean car", "a flat tire"), ("a long nap", "a short night"),
    ("a great book", "a boring lecture"), ("fresh bread", "stale bread"),
    ("my package arriving early", "my package getting lost"), ("a free lunch", "a spoiled lunch"),
    ("a fun weekend trip", "a weekend of chores"), ("a cozy fire", "a leaking roof"),
    ("a sunny morning", "a rainy morning"), ("a good grade", "a bad grade"),
    ("a hot meal", "a cold meal"), ("a quiet night", "a loud night"),
    ("good weather", "bad weather"), ("a happy dog", "a sick dog"),
    ("an early train", "a late train"), ("a clean desk", "a messy desk"),
    # ... and 24 new ones
    ("a surprise party", "a surprise audit"), ("a free coffee", "a spilled coffee"),
    ("a calm flight", "a delayed flight"), ("a picnic in the sun", "a picnic in the rain"),
    ("a quick commute", "a traffic jam"), ("a quiet bus ride", "a crowded bus ride"),
    ("a new phone", "a cracked phone"), ("a good night's sleep", "a car alarm at night"),
    ("a perfect pizza", "a burnt pizza"), ("finding twenty dollars", "losing twenty dollars"),
    ("a week of holidays", "a week of overtime"), ("a smooth presentation", "a frozen laptop"),
    ("an easy exam", "a surprise exam"), ("a lazy beach day", "a painful sunburn"),
    ("a fresh salad", "a stale salad"), ("a working elevator", "a broken elevator"),
    ("a full battery", "a dead battery"), ("a quick doctor visit", "a long doctor visit"),
    ("skiing in fresh snow", "falling on the ice"), ("my flight leaving early", "my flight leaving late"),
    ("a free dessert", "a forgotten order"), ("a romantic date", "getting stood up"),
    ("new shoes", "wet socks"), ("a hot bath", "no hot water"),
]

# positive phrase -> negative versions (order of preference; the first token-matched one is used)
FRAMES = {
    # discovery (18)
    "Wow, I just love that.":            ["Wow, I just hate that.", "Ugh, I just hate that."],
    "This made my day.":                 ["This ruined my day.", "This wrecked my day."],
    "Honestly, it was wonderful.":       ["Honestly, it was awful.", "Honestly, it was terrible.", "Honestly, it was horrible."],
    "I'm so lucky.":                     ["I'm so unlucky.", "I'm so annoyed.", "I'm so upset."],
    "What a perfect day.":               ["What a terrible day.", "What an awful day.", "What a horrible day.", "What a bad day."],
    "Best day ever.":                    ["Worst day ever."],
    "Just perfect.":                     ["Just awful.", "Just terrible.", "Just horrible."],
    "Life is good.":                     ["Life is bad.", "Life is hard.", "Life is cruel."],
    "Absolutely fantastic.":             ["Absolutely terrible.", "Absolutely awful.", "Absolutely horrible."],
    "This is the best.":                 ["This is the worst."],
    "I'm thrilled.":                     ["I'm furious.", "I'm upset.", "I'm miserable.", "I'm annoyed."],
    "What a great way to spend the day.": ["What a bad way to spend the day.", "What an awful way to spend the day.",
                                          "What a terrible way to spend the day."],
    "Yay, so much fun.":                 ["Ugh, so much stress.", "Ugh, no fun at all.", "Ugh, so annoying."],
    "I love my life.":                   ["I hate my life."],
    "Simply amazing.":                   ["Simply awful.", "Simply terrible.", "Simply horrible."],
    "My favorite thing in the world.":   ["My least favorite thing in the world.", "The worst thing in the world."],
    "Nothing could be better.":          ["Nothing could be worse."],
    "Exactly what I wanted.":            ["Exactly what I feared.", "Exactly what I dreaded.", "Not what I wanted."],
    # held-out (6): never used in discovery
    "Great, just what I needed.":        ["Great, just what I feared.", "Ugh, just what I feared.", "Great, that ruined everything."],
    "Living the dream.":                 ["Living the nightmare."],
    "Could this day get any better?":    ["Could this day get any worse?"],
    "Oh, how wonderful.":                ["Oh, how awful.", "Oh, how terrible.", "Oh, how horrible."],
    "Lucky me.":                         ["Poor me.", "Unlucky me."],
    "I couldn't be more thrilled.":      ["I couldn't be more upset.", "I couldn't be more annoyed.", "I couldn't be more disappointed."],
}
HELDOUT_FRAMES = ["Great, just what I needed.", "Living the dream.", "Could this day get any better?",
                  "Oh, how wonderful.", "Lucky me.", "I couldn't be more thrilled."]

CONTEXTS = [  # sentence frames; {s} = situation, {f} = phrase
    "Today was all about {s}. {f}",            # = v1
    "My weekend was all about {s}. {f}",
    "The highlight of my day: {s}. {f}",
    "First thing this morning: {s}. {f}",
]
HELDOUT_CONTEXT = "Guess what my evening looked like: {s}. {f}"
QUESTION = 'Someone said: "{u}"\nIs this person being sarcastic? Answer with Yes or No.'

N_DISC, N_HELD_PHRASE, N_HELD_CONTEXT = 9, 3, 1          # triples per situation pair


def chat_ids(tok, context, s, f):
    """Full chat prompt -> token ids + positions of the situation and the phrase."""
    u = context.format(s=s, f=f)
    q = QUESTION.format(u=u)
    text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    u0 = text.index(u)
    s0 = u0 + u.index(s); s1 = s0 + len(s)
    f0 = u0 + u.rindex(f); f1 = f0 + len(f)
    toks = lambda a, b: {i for i, (x, y) in enumerate(offs) if y > a and x < b}
    return list(ids), {"situation": toks(s0, s1), "frame": toks(f0, f1)}


def aligned(a, b, allowed):
    """Same length, and differences only at the allowed positions."""
    return len(a) == len(b) and all(i in allowed for i, (x, y) in enumerate(zip(a, b)) if x != y)


def build(tok):
    rng = random.Random(SEED)
    disc_frames = [f for f in FRAMES if f not in HELDOUT_FRAMES]
    rows, triple_id = [], 0
    for pair_id, (good, bad) in enumerate(PAIRS):
        plan = ([(f, rng.choice(CONTEXTS), "discovery", "") for f in rng.sample(disc_frames, N_DISC)]
                + [(f, rng.choice(CONTEXTS), "heldout", "phrase") for f in rng.sample(HELDOUT_FRAMES, N_HELD_PHRASE)]
                + [(f, HELDOUT_CONTEXT, "heldout", "frame") for f in rng.sample(disc_frames, N_HELD_CONTEXT)])
        for pf, ctx, split, held in plan:
            S, regS = chat_ids(tok, ctx, bad, pf)
            P, _ = chat_ids(tok, ctx, good, pf)
            ps = aligned(S, P, regS["situation"])
            negs = FRAMES[pf]
            nf, sn = negs[pair_id % len(negs)], False
            for cand in negs:
                if aligned(S, chat_ids(tok, ctx, bad, cand)[0], regS["frame"]):
                    nf, sn = cand, True
                    break
            N, _ = chat_ids(tok, ctx, bad, nf)
            for cond, sit, sit_type, fr, fr_type, ids in [
                ("P", good, "pleasant", pf, "positive", P),
                ("S", bad, "unpleasant", pf, "positive", S),
                ("N", bad, "unpleasant", nf, "negative", N),
            ]:
                text = ctx.format(s=sit, f=fr)
                rows.append(dict(
                    triple_id=triple_id, pair_id=pair_id, condition=cond, text=text,
                    sarcastic=int(cond == "S"), situation=sit, situation_type=sit_type,
                    frame=fr, frame_type=fr_type, positive_frame=pf, context=ctx,
                    prompt=QUESTION.format(u=text), n_tokens=len(ids),
                    token_matched_PS=ps, token_matched_SN=sn, split=split, heldout_type=held))
            triple_id += 1
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    return df


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    df = build(tok)
    df.to_csv(CSV_PATH, index=False)
    print(f"wrote {CSV_PATH}: {len(df)} prompts, {df.triple_id.nunique()} triples")
