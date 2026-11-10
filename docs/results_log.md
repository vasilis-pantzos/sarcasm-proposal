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

### ACDC v2 (`02b_acdc_v2`) — in progress

Changes: 16 + 16 pairs (the 16 pairs of 01b's faithfulness curve), loss = max over experiments of mean |1 − F_i|
(per pair), cap = start loss + 0.02, τ = 0.005, monitor = mean F on 16 + 16 other pairs every 4 receivers
(not used for decisions). Validation on the same kind of unseen pairs as v1, with v1 re-evaluated on them.
Results: `02_acdc/<model>/v2/`.

## Practical notes

* Free Colab: a session stops after inactivity or a maximum duration, and after heavy use no T4 may be available
  for one or more days. All long runs save state on Drive and resume.
