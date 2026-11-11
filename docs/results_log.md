# Results log (for the final report)

Model: **Qwen2.5-7B-Instruct, 4-bit** (bitsandbytes), Colab T4. Development on Qwen2.5-3B-Instruct.
All raw outputs are on Google Drive under `MyDrive/sarcasm/<phase>/<model>/`; the numbers below are copied from
the notebook outputs.

Metric: $d = \log\sum_{Yes,yes} - \log\sum_{No,no}$ at the last position, computed in float32 from the final hidden
state. Recovery / faithfulness: $F = (d_C - d_{corr}) / (d_{clean} - d_{corr})$; clean = S (sarcastic), corrupted
= P (situation swap) or N (phrase swap).

---

## Phase 0 · shared code and checks (`00_setup_check`)

* Checks 2–4 pass (3B).
* Check 1 (batched = single scoring) failed with an absolute tolerance: max |batched − single| = 0.094 in float16,
  caused by the different order of sums with padding, not by a bug. The check now uses a relative tolerance
  (2 % of mean |d|) and the same ranking of sentences.
* Lesson: float16 logits round small differences of d away. **d is now computed in float32** from the final
  hidden state using only the 4 answer rows of the output matrix. This was needed for the 01a validation on 7B.

## Phase 1a · node attribution (`01a_head_attribution`)

* Attribution patching for every head (o_proj input) and MLP, per region (situation, phrase, question, final),
  validated against real activation patching on a subset of pairs (scatter plots on Drive).
* Results: `01a_head_attribution/<model>/head_scores.csv` and figures.

## Phase 1b · EAP-IG on edges (`01b_eap_ig`)

Graph: senders = input, 784 heads, 28 MLPs; receivers = 28 attention blocks, 28 MLP blocks, logits
(at their LayerNorm inputs). **23,577 valid edges.** EAP-IG with 5 interpolation steps.

* Sanity: all edges → exactly clean, no edges → exactly corrupted (after the float32 / nearer-end fix below).
* Agreement with 1a (Spearman, node level): **0.66**.
* Importance lies on the diagonal (sender layer l → receiver l+1): information is passed layer by layer.
  Key zone **layers 11–23, peak 17–19**. Top components: MLPs m14–m23, heads at layer 18 (h4, h13, h15, h18, h21)
  writing into m18; last MLPs m22, m23 write to the logits.

Faithfulness of the top-k edges (16 pairs per experiment; random = k random edges):

| k | edges after pruning | situation | phrase | random |
|---|---|---|---|---|
| ≤ 200 | | 0 | 0 | 0 |
| 1600 | | 0.21 | 0.65 | 0 |
| 3200 | 3200 | 0.50 | **1.07** | 0 |
| 6400 | 6400 | 0.53 | 1.04 | 0 |
| **9600** | 9600 | **1.05** | 1.06 | 0 |
| 12800 | 12800 | 1.01 | 1.10 | 0 |

* Phrase needs ~3200 edges (14 % of the graph), situation 9600 (41 %): the situation information is more
  distributed (longer region, more components). Situation jumps 0.53 → 1.05 between 6400 and 9600.
* F = 0 for small k: an edge only helps if it lies on a complete path input → … → logits; top-k by score is not
  connected. Pruning dangling edges does not change F; it only matters for small k.
* **Candidate circuit for ACDC: top-9600, `candidate_mask.npy`.**

Engineering note (method section): rebuilding a receiver's input from ~800 float16 sender deltas made
"all edges" ≠ clean on 7B (+3.56 vs +3.15). Fix: deltas in float32 with the o_proj weight dequantized once, and
each receiver input rebuilt from the nearer end (x_corr + Σ included, or x_now − Σ excluded).

## Phase 2 · ACDC

Implementation: our own ACDC on the same edge graph. Receivers from the answer backwards; incoming edges ordered
by |EAP-IG score| (weakest first); hierarchical removal (try all incoming edges of a receiver, split in halves on
rejection, down to single edges); receivers that no longer reach the answer lose their edges without a test;
all pairs in one batched forward pass per test (~7 s per test with 16 pairs on a T4).

Data note: the held-out split uses a new phrase whose negative version has a different number of tokens, so it
has **no token-matched phrase pairs**. Phrase is validated on unused discovery pairs only.

### ACDC v1 (`02_acdc`) — overfits

Settings: 8 + 8 pairs, loss = max over experiments |1 − mean F|, τ = 0.01, cap = start loss (0.136) + 0.05.

* Result: **726 edges** (3.1 % of the graph, 7.6 % of the candidate). Two Colab sessions (GPU quota).
* Most kept edges in layers 16–19 (11–29 incoming edges per receiver); layers 24–27 almost empty
  (m27, m26, a25, a24 → 0); logits keep 21 of 260 incoming edges.
* On its **own 16 pairs**: F = 0.851 (situation), 0.815 (phrase): loss ≈ 0.185, i.e. ACDC used the whole margin
  up to the cap.
  * situation per pair: 0.89 0.79 0.81 0.92 0.90 0.85 0.78 0.87 (uniform)
  * phrase per pair: 0.54 1.31 1.11 0.77 1.58 0.96 0.13 0.11 (pairs that break hide behind pairs that overshoot)
  * the candidate on the same pairs: situation 0.98 1.28 1.14 1.20 0.75 0.99 0.58 0.53 (mean 0.931),
    phrase 1.24 1.76 0.90 1.32 0.47 1.42 0.88 1.10 (mean 1.136): per-pair F is noisy even for a good circuit.
* On **pairs it did not see** (situation: 28 held-out + 24 discovery; phrase: 24 discovery):

| circuit | edges | situation | phrase |
|---|---|---|---|
| candidate (EAP-IG top-9600) | 9600 | 0.989 | 1.035 |
| **ACDC v1** | 726 | **0.151** (held-out 0.151, discovery 0.150) | **0.541** |
| EAP-IG top-726 | 726 | 0.066 | 0.115 |
| random 726 inside the candidate | 726 | 0.000 | 0.000 |

* Conclusions: (1) at the same size ACDC > EAP-IG > random: ACDC assembles connected paths; (2) with 8 pairs per
  experiment ACDC memorises the routes of those pairs (situation is fine on all 8 training pairs, 0.15 on new
  ones); (3) a loss on the mean lets broken pairs be compensated by overshooting ones; (4) the minimal circuit that
  generalises lies between 726 and 9600 edges.

### ACDC v2 (`02b_acdc_v2`) — stopped early, overfits too

Changes: 16 + 16 pairs (the 16 pairs of 01b's faithfulness curve), loss = max over experiments of mean |1 − F_i|
(per pair), cap = start loss + 0.02, τ = 0.005, monitor = mean F on 16 + 16 other pairs every 4 receivers
(not used for decisions). Ran on an A100: 1.8 s per test with 32 pairs (T4 with 16 pairs: ~7 s).

* Sanity: candidate mean F on the 16 + 16 pairs 1.048 / 1.031 vs 1.045 / 1.055 in 01b (T4); per pair up to 0.3
  apart (other GPU, other batch) → the check compares means with tolerance 0.05.
* Start loss (per pair) 0.333, cap 0.36. Candidate on the monitor pairs: situation 1.006, phrase 0.946.
* After 8 receivers (logits → a24; logits 260 → 12 edges, m27 → 2, m26 → 4, m24 → 2):
  loss on its own pairs 0.333 → 0.315 (flat or better), but **monitor F 0.836 / 0.717 (4 receivers) →
  0.770 / 0.642 (8 receivers)**. Stopped.
* Why: per-pair F is noisy (≈ 0.3 even for the candidate) → with 16 pairs per experiment the loss is uncertain
  by ≈ 0.08, far more than τ = 0.005 per decision; thousands of edge-by-edge decisions fit that noise.
  The partial state is `02_acdc/<model>/v2/acdc_state_tau0.005_n16_per_pair.pt`.

### ACDC v3 (`02c_acdc_v3`) — one cut-off per receiver; stopped after 13 receivers

* Order of the incoming edges of each receiver = |EAP-IG score| (from all discovery pairs). ACDC only decides
  **how many** of the strongest to keep: k = 0 first, then binary search on k (≈ log2(n) tests) → ≈ 57 decisions.
* Pairs: all correct discovery pairs except 16 + 16 (monitor) and 24 + 24 (validation), at most 64 per
  experiment; loss per pair, cap = start + 0.02, τ = 0.005. A100: ~1 min per receiver.
* Sanity on the ACDC pairs: candidate mean F 1.053 / 1.044 vs 1.045 / 1.055 in 01b. Start loss 0.247.
* Kept per receiver (k of n): logits 18/260, m27 15/196, a27 7/129, m26 50/160, a26 19/116, m25 43/141,
  a25 0/72, m24 0/141, a24 4/66, m23 137/163, a23 7/107, m22 22/176, a22 30/129. Loss 0.247 → 0.270.
* Monitor (candidate: situation 0.991, phrase 0.927):

| receivers done | situation | phrase |
|---|---|---|
| 4 | 0.892 | 0.850 |
| 8 | 0.748 | 0.851 |
| 12 | 0.733 | 0.874 |

* Phrase generalises (stable ≈ 0.85-0.87, compare v2: 0.72 → 0.64); situation does not: the big step is when the
  whole m24 (and a25) is removed, which leaves the loss on the ACDC pairs unchanged (0.257 → 0.258) but drops the
  monitor 0.89 → 0.75. Likely cause: situation has far fewer correct token-matched pairs than phrase, and 40 of
  them were set aside for monitor + validation. Stopped.

### ACDC v3b (`02d_acdc_v3b`) — more situation pairs, zero tolerance

* Situation validated on the 28 held-out pairs only (phrase on 24 unused discovery pairs); monitor 8 + 8 pairs;
  all other correct discovery pairs (≤ 64 per experiment) go to ACDC.
* Run 1: **τ = 0, cap = start loss** (an edge goes only if the circuit does not get worse); run 2: τ = 0.005,
  cap = start + 0.02, starting from run 1's circuit.
* Results: `02_acdc/<model>/v3b/`. Planned final comparison: EAP-IG candidate, ACDC v1, ACDC v3b (both runs).

**Run 1 (τ = 0, cap = start loss), L4, 446 tests, 50 min:** 9600 → **7078 edges** (30 % of the graph, 74 % of the
candidate). Loss on the ACDC pairs 0.241 → **0.198** (never worse; removing edges made pairs *more* faithful).

Monitor (8 + 8 pairs, ±0.1; candidate: situation 0.941, phrase 0.881), every 4 receivers:
situation 0.805, 0.841, 0.855, 0.879, 0.913, 0.906, 0.913, 0.921, 0.946, 0.966, 1.031, 0.974, 0.984, 0.984;
phrase 0.791, 0.829, 0.893, 0.931, 0.868, 0.804, 0.808, 0.816, 0.918, 0.894, 0.850, 0.898, 0.854, 0.854.
→ **no loss of generalisation** (v3: situation 0.73 after 12 receivers).

Kept per receiver (k of n):
* cut hard: logits 19/260, a27 0/129, a25 0/72, a4 0/44, m22 20/176, a21 21/101, m21 41/173, a22 44/129,
  m18 77/275, a12 109/221, a5 17/46, m4 49/86, m6 78/121;
* kept (almost) all: m26 158/160, m23 161/163, m19 248/248, m16 344/344, a17 290/311, a16 282/288, a15 302/305,
  m14 337/337, a14 281/281, m13 304/304, a13 245/245, layers 7–11 and 0–3 all or nearly all.
→ With zero tolerance the middle of the network (layers ≈ 7–20) is **dense**: almost every edge contributes a
little; edges are only removable near the answer (layers 21–27, logits) and in a few early blocks.

**Run 2 (τ = 0.005, cap = start + 0.02, from run 1):** 7078 → ≈ 5100 edges; loss on the ACDC pairs 0.198 → ≈ 0.26
(at the cap); monitor situation 0.996 → 0.895 (slow decline), phrase 0.80–0.98 (noise).

**Validation** (situation: 28 held-out pairs; phrase: 24 unused discovery pairs; values read from the figure,
exact table to be added). Final selection: τ = 0, **7078 edges** (30.0 % of the graph, 73.7 % of the candidate);
target (0.9 ≤ F ≤ 1.2 in both) not reached by any ACDC circuit.

| circuit | edges | situation (held-out) | phrase |
|---|---|---|---|
| candidate | 9600 | ≈ 0.92 | ≈ 1.00 |
| ACDC v1 | 726 | ≈ 0.15 | ≈ 0.50 |
| ACDC v3b τ = 0 | 7078 | ≈ 0.71 | ≈ 0.89 |
| EAP-IG top-7078 | 7078 | ≈ 0.71 | ≈ 0.92 |
| ACDC v3b τ = 0.005 | ≈ 5100 | ≈ 0.70 | ≈ 0.79 |
| EAP-IG top-5100 | ≈ 5100 | ≈ 0.70 | ≈ 1.14 |
| random, same size | | ≈ 0 | ≈ 0 |

**Key finding: the held-out split is a test on a NEW PHRASE.** All held-out triples use "Great, just what I
needed.", which never appears in discovery (discovery: 4 phrases, 46 triples each). For situation:
monitor (discovery pairs, known phrases) ≈ 0.98 at the end of run 1, held-out (new phrase) ≈ 0.71, while the
candidate gives ≈ 0.94 / 0.92. → Pruned circuits generalise to new *situations* with the known phrases, but lose
part of the generalisation to a new *phrase*. (In v1 both were 0.15: there the circuit failed everywhere.)

Other conclusions: at small sizes ACDC ≫ EAP-IG top-k (connected paths); at large sizes (5000–7000 edges) top-k
is already connected and ACDC adds nothing. No circuit smaller than the candidate keeps F ≥ 0.9 in both
experiments: the mechanism is distributed (dense core in layers ≈ 7–20; few paths near the answer).

## Phase 2e · circuits answering alone (`02e_compare_circuits`)

Every model on all 690 prompts, with edge-level **mean ablation** (edges outside the circuit get the dataset mean,
per position, over prompts of the same length): full model, candidate, ACDC v1, v3b τ = 0, v3b τ = 0.005,
random (size of v3b τ = 0), no edges. Measures per split: Yes rate per condition, accuracy, triple accuracy,
agreement with the full model, correlation of d. Results: `02e_compare_circuits/<model>/`.

## Dataset v2 (`data/make_dataset_v2.py`, `notebooks/data_v2.ipynb`)

Why: the circuits found on v1 generalise to new situations with the known phrases, partly to the new phrase, and
poorly to free text (own sentences: e.g. "My house burned down. This is the happiest day of my life": full model
Yes, d moves +5.8 from the average prompt; v3b circuits 8 %, −3 %, −41 % of that move). v1 has 5 phrases and a
single sentence frame ("Today was all about {s}. {f}").

| | v1 | v2 |
|---|---|---|
| situation pairs | 46 | 70 (24 new; 29 reworded so that both versions have the same tokens) |
| positive phrases | 5 (1 held out) | 24 (6 held out), each with token-matched negative versions |
| sentence frames | 1 | 5 (1 held out: "Guess what my evening looked like: {s}. {f}") |
| triples | 230 | 910: 630 discovery, 210 held-out phrase, 70 held-out frame |
| token-matched (P–S / S–N) | 116 / 184 of 230; held-out phrase: 0 S–N | **all 910 / 910** |

Prompt length 56–67 tokens. The tokenizer used to design it (qwen.tiktoken) reproduces v1 exactly (all 690 lengths,
all 230 + 230 token-matched flags).

Plan: (1) full model + v1 circuits on v2 (02e, `DATA = "v2"`, `CIRCUITS_FROM = "v1"`); (2) EAP-IG (01b) and ACDC v3b
(02d) on v2; (3) 02e with the v2 circuits.

### Step 1: the v1 circuits on v2 (`02e`, DATA = v2, CIRCUITS_FROM = v1)

2730 prompts; groups: discovery (1890), held-out phrase (630), held-out frame (210). For the v1 circuits all of v2
is new except the 4 discovery phrases and the v1 sentence frame. (Triple accuracy / agreement read from the figure.)

| | triple acc. disc / phrase / frame | agreement disc / phrase / frame | F alone (slope) disc / frame / phrase | bal. acc. (d>0) disc / phrase / frame |
|---|---|---|---|---|
| full model | 0.81 / 0.79 / 0.84 | 1 | 1 | 0.783 / 0.737 / 0.814 |
| EAP-IG candidate (9600) | 0.78 / 0.77 / 0.84 | 0.90 / 0.90 / 0.78 | 0.818 / 0.698 / 0.813 | 0.752 / 0.711 / 0.704 |
| ACDC v3b τ=0 (7078) | 0.78 / 0.87 / 0.86 | 0.87 / 0.86 / 0.74 | 0.683 / 0.535 / 0.608 | 0.737 / 0.690 / 0.714 |
| ACDC v3b τ=0.005 (≈5100) | 0.78 / 0.79 / 0.81 | 0.83 / 0.81 / 0.76 | 0.497 / 0.412 / 0.384 | 0.729 / 0.677 / 0.775 |
| ACDC v1 (726) | 0.59 / 0.49 / 0.49 | 0.47 / 0.62 / 0.45 | −0.008 / −0.079 / 0.128 | 0.565 / 0.538 / 0.529 |
| random (7078) | 0.33 / 0.22 / 0.34 | 0.61 / 0.56 / 0.47 | 0.016 / 0.067 / −0.054 | 0.500 / 0.500 / 0.500 |
| no edges | 0 / 0 / 0.02 | 0.53 / 0.60 / 0.43 (baseline) | – | 0.500 |

Median F alone (disc / frame / phrase): EAP-IG 0.848 / 0.736 / 0.842; v3b τ=0 0.755 / 0.584 / 0.644;
τ=0.005 0.591 / 0.432 / 0.408; v1 −0.189 / −0.196 / 0.177; random 0.217 / 0.252 / −0.117.
Own thresholds (calibrated on v2 discovery) change balanced accuracy by ≤ 0.07.

* The full model is slightly worse on v2 (triple accuracy 0.79–0.84 vs 0.85 on v1) but still good.
* **The ranking generalises**: circuits found with 4 phrases and 1 frame keep the full model's triple accuracy on
  20 new phrases and 4 new frames (EAP-IG, v3b τ=0 and τ=0.005: 0.77–0.87).
* **The size of the effect does not fully**: F alone drops from 0.91 / 0.78 / 0.65 (v1 data) to 0.82 / 0.68 / 0.50
  (v2 discovery) and further on the new frame (0.70 / 0.54 / 0.41); agreement drops most on the new frame
  (0.74–0.78). The sentence frame (syntax) hurts more than a new phrase.
* v1 ACDC (726) ≈ random in F alone; random and no-edges behave as expected.

Faithfulness with corrupted partners (pairs; 32 correct token-matched pairs per cell; columns in the order of the
circuits; first time with token-matched **phrase** pairs on held-out phrases):

| group | experiment | EAP-IG (9600) | ACDC v1 (726) | v3b τ=0 (7078) | v3b τ=0.005 | random |
|---|---|---|---|---|---|---|
| discovery | situation | 1.001 | 0.268 | 0.799 | 0.690 | 0.001 |
| discovery | phrase | 0.823 | 0.484 | 1.024 | 1.052 | 0.009 |
| held-out phrase | situation | 0.912 | 0.193 | 0.781 | 0.625 | −0.000 |
| held-out phrase | phrase | 0.740 | 0.276 | 0.743 | 0.716 | −0.002 |
| held-out frame | situation | 1.078 | 0.165 | 0.846 | 0.715 | −0.002 |
| held-out frame | phrase | 0.803 | 0.343 | 0.760 | 0.711 | 0.002 |

* With the measure used to find them (patching from the partner), the v1 circuits stay faithful on v2: situation
  0.78–1.08 for EAP-IG and v3b τ=0, even with the new frame. The **phrase swap on new phrases** is the weak spot
  (≈ 0.72–0.74 for all three large circuits). v1 ACDC (726) 0.17–0.48, random 0.
* Pair faithfulness is higher than F alone (mean ablation): answering alone is the harder test.

### Step 2: new circuits on v2

**EAP-IG (01b, DATA = v2):** 200 correct discovery pairs per experiment (seeded shuffle, so they cover all situation
pairs, phrases and frames). Faithfulness of top-k (16 pairs, read from the figure): situation 1600: 0.27, 3200: 0.78,
6400–12800: ≈ 0.90; phrase 800: 0.47, 1600: 1.06, 3200: 1.43, 6400: **1.72** (CI up to 2.3), 9600: 1.13, 12800: 1.29.
The phrase swap **overshoots** strongly (some phrase pairs have small margins d(S) − d(N) ≈ 1, so F is unstable).
The old rule "smallest k with F ≥ 0.9" picked 6400; with the new rule (0.9 ≤ F ≤ 1.2, else closest to 1) the
candidate is **9600 edges**, as on v1 (k comes from a fixed grid, so the same size is expected; the edges differ).

**ACDC v3b (02d, DATA = v2), L4:** ACDC pairs: all correct discovery pairs (shuffled) except monitor (8 + 8) and
validation, ≤ 64 per experiment → 128 pairs, 8.3 s per test. Start loss (per pair) 0.228; mean F of the candidate on the
ACDC pairs: situation 0.858, phrase 1.065. Sanity: all / no edges exact; "same F as 01b" not comparable (01b and 02d
kept different correct pairs on different GPUs → different shuffled 16; per-pair noise up to 1.2 on phrase pairs).

Final circuit: **τ = 0, 8147 edges** (34.6 % of the graph, 84.9 % of the candidate). Validation (situation: 64 held-out
pairs, new phrases + new frame; phrase: 64 held-out + 24 unused discovery; read from the figure):

| circuit | edges | situation | phrase |
|---|---|---|---|
| candidate | 9600 | ≈ 0.86 | ≈ 1.07 |
| **ACDC v3b τ = 0** | **8147** | **≈ 0.81** | **≈ 0.99** |
| EAP-IG top-8147 | 8147 | ≈ 0.76 | ≈ 1.05 |
| ACDC v3b τ = 0.005 | ≈ 5700 | ≈ 0.75 | ≈ 0.84 |
| EAP-IG top-5700 | ≈ 5700 | ≈ 0.74 | ≈ 1.07 |
| random | | ≈ 0 | ≈ 0 |

* Compared with v1 (τ = 0: 7078 edges, situation 0.71 on one new phrase, phrase 0.89 on known phrases only), the v2
  circuit is more faithful on harder held-out data (new phrases **and** a new frame; phrase now tested on new phrases).
* On v2 **ACDC beats EAP-IG at the same size** (situation 0.81 vs 0.76; phrase 0.99 vs an overshooting 1.05); on v1
  they were equal at these sizes. The variety of v2 lets ACDC keep the right edges.
* Slightly larger than the v1 circuit (8147 vs 7078): covering 18 phrases and 4 frames needs more of the mechanism.
* No circuit reaches 0.9 on the situation swap with new phrases / frames (candidate 0.86).

### Step 3: the v2 circuits on v2 (`02e`, DATA = v2, CIRCUITS_FROM = v2) and the comparison with the v1 circuits

Same prompts and pairs as step 1, so the two sets of circuits are compared on identical data
(v1 circuits → v2 circuits; read from the figures where marked ≈).

Triple accuracy (disc / new phrase / new frame; ≈): EAP-IG 0.78 / 0.77 / 0.84 → 0.79 / 0.88 / 0.77;
ACDC τ=0 0.78 / 0.87 / 0.86 → 0.77 / 0.83 / 0.67; τ=0.005 0.78 / 0.79 / 0.81 → 0.72 / 0.75 / 0.58.
Agreement with full (≈): EAP-IG 0.90 / 0.90 / 0.78 → 0.88 / 0.89 / 0.77; τ=0 0.87 / 0.86 / 0.74 → 0.88 / 0.87 / 0.78;
τ=0.005 0.83 / 0.81 / 0.76 → 0.85 / 0.85 / 0.69.

F alone (slope; disc / new frame / new phrase):

| circuit | edges v1 → v2 | v1 circuits | v2 circuits |
|---|---|---|---|
| EAP-IG candidate | 9600 → 9600 | 0.818 / 0.698 / 0.813 | 0.760 / 0.611 / 0.707 |
| ACDC v3b τ=0 | 7078 → 8147 | 0.683 / 0.535 / 0.608 | 0.735 / 0.565 / 0.679 |
| ACDC v3b τ=0.005 | ≈5100 → ≈5700 | 0.497 / 0.412 / 0.384 | 0.610 / 0.478 / 0.568 |
| random | | ≈ 0 | 0.014 / 0.054 / −0.047 |

(v2 medians: τ=0 0.770 / 0.694 / 0.670; τ=0.005 0.658 / 0.450 / 0.586; EAP-IG 0.796 / 0.765 / 0.685.)
Balanced accuracy, d > 0 (disc / new phrase / new frame), v2 circuits: EAP-IG 0.727 / 0.696 / 0.675; τ=0 0.729 /
0.675 / 0.671; τ=0.005 0.722 / 0.686 / 0.625 (v1 circuits: 0.752 / 0.711 / 0.704; 0.737 / 0.690 / 0.714; 0.729 / 0.677
/ 0.775). Own thresholds change ≤ 0.07.

Faithfulness with corrupted partners (32 pairs per cell; v1 circuits → v2 circuits):

| group | experiment | EAP-IG | ACDC τ=0 | ACDC τ=0.005 | random (v2) |
|---|---|---|---|---|---|
| discovery | situation | 1.001 → 0.827 | 0.799 → 0.790 | 0.690 → 0.708 | −0.002 |
| discovery | phrase | 0.823 → 1.244 | 1.024 → 1.219 | 1.052 → 1.099 | 0.006 |
| held-out phrase | situation | 0.912 → 0.879 | 0.781 → 0.778 | 0.625 → 0.695 | 0.001 |
| held-out phrase | **phrase** | **0.740 → 1.133** | **0.743 → 1.037** | 0.716 → 0.766 | 0.003 |
| held-out frame | situation | 1.078 → 0.945 | 0.846 → 0.869 | 0.715 → 0.873 | −0.000 |
| held-out frame | **phrase** | **0.803 → 0.984** | **0.760 → 0.939** | 0.711 → 0.762 | 0.002 |

**Conclusions of the v1 / v2 comparison**
1. The specific weak spot of the v1 circuits, the **phrase swap on new phrases / a new frame** (≈ 0.74–0.80), is
   fixed by the v2 circuits under pair patching (ACDC τ=0: 1.04 on new phrases, 0.94 on the new frame). Learning
   from 18 phrases and 4 frames captures how the model reads the *phrase* in general. (Discovery phrase pairs now
   overshoot, 1.1–1.24: v2's phrase pairs include small-margin pairs, where F is unstable.)
2. When a circuit answers **alone** (mean ablation), the v2 data do not help beyond size: ACDC v2 circuits are a bit
   better but also bigger; the v2 EAP-IG candidate (same 9600 edges) is *worse* than v1's. Ordered by size, F alone
   follows the number of edges (≈5100: 0.50, ≈5700: 0.61, 7078: 0.68, 8147: 0.74, 9600: 0.76–0.82), not the dataset.
3. Triple accuracy / balanced accuracy: similar; the v2 ACDC circuits are *worse* on the new frame (0.67 vs 0.86 at
   τ=0; 70 triples, ±≈0.05–0.1).
4. Overall: more varied data give circuits that transfer the **phrase** part of the computation better, but no small
   circuit that answers like the model on its own. Faithfulness alone grows with circuit size: the mechanism is
   **distributed**, and this is a property of the model, not of the dataset.

### Overlap of the v1 and v2 circuits (edges)

| circuit | edges v1 | edges v2 | shared | Jaccard | share of the smaller | random expected | shared / random |
|---|---|---|---|---|---|---|---|
| EAP-IG candidate | 9600 | 9600 | 7919 | 0.702 | 0.825 | 3909 | 2.03 |
| ACDC v3b τ=0 | 7078 | 8147 | 5549 | 0.573 | 0.784 | 2446 | 2.27 |
| ACDC v3b τ=0.005 | 5062 | 5722 | 3805 | 0.545 | 0.752 | 1229 | 3.10 |

EAP-IG edge scores v1 vs v2: Spearman **0.868**; top-100 edges 87 % shared, top-500 79 %, top-2000 78 %
(random: 0.4 %, 2.1 %, 8.5 %). Per sender layer (ACDC τ=0, `circuits_v1_vs_v2_per_layer.png`): same profile
(most edges from layers 1–12, decreasing to the answer); v2 keeps **more edges from layers 14–27** and the layer-4
components (v1 had removed a4 entirely, so its layer-4 heads were dangling).

→ **The same mechanism**: ≈ 75–83 % of each circuit is shared, 2–3 × above chance, and the strongest edges agree
most (87 % of the top-100). v2 adds edges mainly in the later layers, consistent with its better transfer of the
phrase swap to new phrases and frames.

## Phase 3a · sarcasm direction (`03a_sarcasm_direction`, DATA = v2, full model)

Residual stream at the last token, all layers, 2730 prompts (+ 910 for the fourth condition Q).

**Probes per layer** (L2 logistic regression, trained on discovery; held-out balanced accuracy):
* phrase polarity ≈ 1.0 from layer 1; situation valence high early (dips on the new frame in layers 6–12);
* **sarcasm generalises to new phrases / the new frame only from layer ≈ 14–19**: best layer **19**
  (held-out phrase 0.963, held-out frame 0.989). (Discovery = training data, ≈ 1.0 everywhere: not informative.)

**More than sentiment? — 2 × 2 design.** With only P, S, N the sarcasm mean-difference is algebraically a combination
of the phrase and situation mean-differences (checked: residual 1e-8), so "remove the sentiment directions" gives
chance by construction; this first version of the test was dropped. Adding **Q = pleasant situation + negative
phrase** (P's sentence with N's phrase):
* the full model answers **Yes on Q** in 53 % (discovery), 72 % (new phrases), 86 % (new frame) of prompts;
* interaction ι = (S − N) − (P − Q) relative to the main effects |ι| / (|φ| + |σ|): ≈ 0 up to layer 10, rising, then a
  **jump at layers 17–21 to ≈ 2** (1.42 at layer 19);
* cos(ι on discovery, ι on held-out): ≈ 0.85 by layer 9, **0.97–0.99 from layer 14** (new phrases and new frame);
* probe S vs (P, N, Q) after removing the phrase and situation directions: ≈ 0.6 (layer 19: 0.625 / 0.593 / 0.486):
  the interaction is shared by S and Q, so it does not single out S.

→ The model computes a **situation–phrase incongruity** signal (a non-additive interaction), general across phrases
and frames, appearing at layers ≈ 17–21, and it treats incongruity in either direction as "sarcastic".

**Steering** (add α · (mean S − mean P/N) at the last token, held-out prompts; Yes rates read from the figure):
layers 6–16: no effect; 18: small; **20–24: full control** — P 0.15 → ≈ 0.5–0.9 (α = 1) → ≈ 1.0 (α = 2);
N 0.5 → 0.9–1.0; S 0.86 → 0.35–0.5 (α = −1) → 0 (α = −2). Random direction of the same norm (layer 19): little effect.
(The full model says Yes on ≈ 50 % of held-out N.)

**Who writes the direction (read at layer 19; share of the S − P/N gap):** m18 **0.611**, a18.h15 0.114, a18.h18 0.071,
a18.h4 0.035, m17 0.027, a18.h13 0.026, a18.h16 0.024, a18.h20 0.017, a17.h1 0.014, a18.h21 0.013 (≈ 95 % together);
then small contributions from layers 15–18 (a18.h17 −0.012, a18.h19 −0.007, a18.h7 −0.006 push the other way).
**All 25 top writers are in both ACDC circuits (v1 and v2)**, and the five layer-18 heads found by EAP-IG in Phase 1b
(h4, h13, h15, h18, h21) are all in the top 10. Sanity: embedding + heads + MLPs = residual stream (exact).

**Convergence of methods on layers ≈ 17–20:** EAP-IG (heads layer 18 → m18), ACDC (dense core 14–20), generalising
probe (from 14–19), incongruity interaction (jump 17–21), writers (m18 + layer-18 heads ≈ 95 %), steering (from 20).
The final computation is *localised* (m18 and a few layer-18 heads write the decision), while the inputs (the two
sentiments) are computed in many places earlier, which is why edge-level circuits are large.

## Practical notes

* Free Colab: a session stops after inactivity or a maximum duration, and after heavy use no T4 may be available
  for one or more days. All long runs save state on Drive and resume.
