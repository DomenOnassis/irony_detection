# Slovene Irony Pipeline (separate workspace module)

This folder is fully separate from the original `subtaskA` and `subtaskB` code.

It provides Python scripts to:
- parse SemEval-style Slovene files,
- train Task A (binary irony) and Task B (multi-class irony),
- run prediction on test sets,
- keep a Word2Vec API-compatible loading layer you can later replace,
- keep an API feature hook you can later connect to your existing external APIs.

## Main scripts

- `train_taskA.py`
- `train_taskB.py`
- `predict_taskA.py`
- `predict_taskB.py`
- `run_all.py`
- `api_hooks.py`

## Notes

- The current feature stack uses:
  - n-gram TF-IDF,
  - mean Word2Vec sentence vectors,
  - optional emoji2vec vectors.
- Keep your current APIs and model files for now, then replace paths later.
- Datasets default to `datasets/slovene/...`.

## Quick start

**You already have the Slovene W2V model at:**
```
D:\FERI - MAG\1. letnik\2. semester\JT\all-token-prelim.ft.sg.bin
```

So just run:
```bash
pip install -r slovene_pipeline/requirements.txt
cd slovene_pipeline
python run_all.py
```

No need to pass paths—defaults are pre-configured.

## Example with custom paths

```bash
cd slovene_pipeline
python train_taskA.py --w2v-path path/to/custom.bin --w2v-binary
```

## API hooks

`api_hooks.py` currently returns zero-width feature matrices (no-op).
Replace `build_api_features(...)` internals with your current API pipeline and keep the same function signature.
