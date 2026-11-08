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
data/sarcasm_dataset.csv     230 triples (690 sentences), discovery / held-out split
src/                         shared code used by every notebook
  model.py                   load Qwen2.5-3B (development) or 7B (final, 4-bit)
  prompts.py                 dataset rows -> token ids + regions; clean/corrupted pairs
  metric.py                  d = logit(Yes) - logit(No); batched scoring; triple accuracy
  hooks.py                   read / replace activations (residual, attention, MLP, per head)
  patching.py                activation patching and the recovery metric R
  utils.py                   results on Google Drive; compute-once caching
notebooks/
  data.ipynb                 builds and describes the dataset           (pilot)
  eval.ipynb                 compares four Qwen models                  (pilot)
  patching_7b.ipynb          activation patching on Qwen2.5-7B          (pilot)
  00_setup_check.ipynb       checks that src/ reproduces the pilot results
results/                     outputs of the pilot notebooks
```

## How to run

Open a notebook in Google Colab with a **T4 GPU**. The first cell clones this repository and
imports `src/`; results are saved to Google Drive (`MyDrive/sarcasm/results`), so an interrupted
run continues where it stopped. New code is developed on Qwen2.5-3B-Instruct and the final results
are produced on Qwen2.5-7B-Instruct.

## Work plan

| phase | task | status |
|---|---|---|
| 0 | shared code and checks | done |
| 1 | head-level attribution, then EAP-IG on edges | |
| 2 | minimal circuit with ACDC | |
| 3 | path patching and attention patterns | |
| 4 | faithfulness, minimality, random circuits, held-out phrase | |
| 5 | implicit question ("Did the person enjoy their day?") | |
| 6 | iSarcasmEval | |
