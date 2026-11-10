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

## Practical notes

* Free Colab: a session stops after inactivity or a maximum duration, and after heavy use no T4 may be available
  for one or more days. All long runs save state on Drive and resume.
