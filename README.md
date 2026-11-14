# When Words and Situations Disagree: Finding the Sarcasm Circuit in an LLM

Research project for the course "Advanced Topics in Artificial Intelligence" (NTUA, 2026–2027).

We look for the components of Qwen2.5-7B-Instruct that detect one well-defined type of sarcasm
(positive words about a clearly negative situation), using controlled triples of sentences:

| | situation | phrase | sarcastic |
|---|---|---|---|
| P | pleasant | positive | no |
| S | unpleasant | positive | **yes** |
| N | unpleasant | negative | no |

## Repository

```
data/sarcasm_dataset.csv     v1: 230 triples (690 sentences), 5 phrases, 1 sentence frame
data/sarcasm_dataset_v2.csv  v2: 910 triples (2730 sentences), 24 phrases, 5 sentence frames, all token-matched
data/make_dataset_v2.py      builds v2 (held-out: 6 phrases + 1 sentence frame never seen in discovery)
src/                         shared code used by every notebook
  model.py                   load Qwen2.5-3B (development) or 7B (final, 4-bit)
  prompts.py                 dataset rows -> token ids + regions; clean/corrupted pairs
  metric.py                  d = logit(Yes) - logit(No); batched scoring; triple accuracy
  hooks.py                   read / replace activations (residual, attention, MLP, per head)
  patching.py                activation patching and the recovery metric R
  attribution.py             attribution patching for heads and MLPs (gradient estimate of R)
  eap.py                     EAP-IG edge scores, edge patching, faithfulness of circuits
  acdc.py                    batched edge patching and hierarchical ACDC (minimal circuit)
  standalone.py              a circuit answering single prompts on its own (edge-level mean ablation)
  direction.py               residual stream per layer, probes, steering, components writing a direction
  flow.py                    attention of the last token per head and region; what heads write, by source region
  pathpatch.py               pattern vs value patching, knockout at the last token, route patching
  utils.py                   results on Google Drive; compute-once caching
notebooks/
  data.ipynb                 builds and describes the dataset           (pilot)
  data_v2.ipynb              builds and describes dataset v2 (no GPU)
  eval.ipynb                 compares four Qwen models                  (pilot)
  patching_7b.ipynb          activation patching on Qwen2.5-7B          (pilot)
  00_setup_check.ipynb       checks that src/ reproduces the pilot results
  01a_head_attribution.ipynb attribution for every head/MLP, validated against real patching
  01b_eap_ig.ipynb           EAP-IG on edges, faithfulness of the top-k edges -> candidates for ACDC
  02_acdc.ipynb              ACDC v1 (8+8 pairs): overfits, kept for the report
  02b_acdc_v2.ipynb          ACDC v2 (16+16 pairs, per-pair loss, monitor): still overfits, stopped
  02c_acdc_v3.ipynb          ACDC v3: one cut-off per receiver (EAP-IG order): situation still overfits, stopped
  02d_acdc_v3b.ipynb         ACDC v3b: more situation pairs, zero tolerance, then tau = 0.005 -> the circuit
  02e_compare_circuits.ipynb every circuit answers alone (mean ablation) on all prompts vs the full model
  03a_sarcasm_direction.ipynb probes per layer, 2x2 incongruity test, steering, who writes the direction, own sentences
  03b_information_flow.ipynb region patching per layer, attention per head in P/S/N/Q, what key heads read and write
  03c_path_patching.ipynb    switch (pattern vs value), knockout, routes through phrase/question tokens, mover heads
docs/results_log.md          numbers and findings of every phase, for the final report
results/                     outputs of the pilot notebooks
```

## How to run

Open a notebook in Google Colab with a **T4 GPU**. The first cell clones this repository and
imports `src/`; results are saved to Google Drive (`MyDrive/sarcasm/results`), so an interrupted
run continues where it stopped. New code is developed on Qwen2.5-3B-Instruct and the final results
are produced on Qwen2.5-7B-Instruct.

Notebooks 01b, 02d and 02e have a setting `DATA = "v1"` / `"v2"`; with v2 the results go to separate folders
(`..._data-v2`), so the v1 results are kept. 02e also has `CIRCUITS_FROM` (test v1 circuits on v2 data).

## Work plan

| phase | task | status |
|---|---|---|
| 0 | shared code and checks | done |
| 1a | head-level attribution, validated against real patching | done |
| 1b | EAP-IG on edges | done |
| 2 | minimal circuit with ACDC (+ dataset v2) | done |
| 3 | what the model represents: sarcasm direction (3a), information flow (3b), path patching (3c) | in progress |
| 4 | faithfulness, minimality, random circuits, held-out phrase | |
| 5 | implicit question ("Did the person enjoy their day?") | |
| 6 | iSarcasmEval | |
