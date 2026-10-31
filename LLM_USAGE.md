# LLM usage

Tool: Claude Code (command-line coding assistant) running Claude Sonnet 5.5, in a single working session on 2026-10-30 / 2026-10-31. This file was drafted by the assistant from that session; the authors should review and amend it.

## What the assistant did

| Area | Work | Files |
|---|---|---|
| Repository review | Read every file, summarised each, listed Windows/Keras-3 incompatibilities (Colab data path, `Adam(lr=...)`, `input_shape` on layers, `plt.show()` before `plt.savefig()`) | none changed |
| Environment | Created `.venv` (Python 3.12), installed tensorflow, pyedflib, scikit-learn, matplotlib, numpy, pandas, seaborn, pinned versions | `requirements.txt` |
| Data pipeline | Configurable data path, download of the first 20 Sleep-Cassette subjects, subject-wise 70/10/20 split (seed 42), per-recording `.npz` cache, class counts per split | `config.py`, `data.py`, `run_pipeline.py`, `training.py` |
| Training script | Trains MLP / SimpleRNN / 1D-CNN on the same split with early stopping, class weights, RNN time-downsampling decision, per-model outputs | `train_all.py`, `models.py` |
| v2 setup | Per-recording z-score normalisation with a versioned cache, `--lr`, `--patience`, `--run-name`, `--no-normalise`, `--no-reduce-lr`, ReduceLROnPlateau | `data.py`, `models.py`, `train_all.py`, `.gitignore` |
| Evaluation | Test-set metrics, comparison tables, confusion matrices, per-class F1 chart, error-analysis notes | `evaluate.py` |
| Documentation | README sections for files, results, how to run, acknowledgements; this file | `README.md`, `LLM_USAGE.md` |

## What the assistant did not do

- It did not choose the architectures; the layer definitions in `models.py` are the authors' (only deprecated arguments and the learning-rate parameter were changed).
- It did not run the full training runs. The authors ran `train_all.py` for v1 and v2; the assistant only ran short smoke tests (1 epoch, 2-5% of the data).
- Model results in `results/` come from the authors' runs. The assistant ran `evaluate.py` on them and copied the printed numbers into `README.md`.
- No report text was written by the assistant.

## How outputs were checked

- Each pipeline stage was run on real data before being reported as working: a 3-subject download for the data pipeline, then 1-epoch smoke tests on 2% (3 subjects) and 5% (20 subjects) of the data for `train_all.py`.
- The test-label MD5 (`9e5ab0415363e4074ad14c729bab8f50`) is printed by `train_all.py`, stored in every `run_config.json`, and re-checked by `evaluate.py` for each evaluated run, together with the input shape implied by the run's preprocessing.
- The cached z-scored arrays were checked against the raw ones (per-channel mean about 0 and a different scale), and the v1 path was checked to reuse the raw cache.
- Not verified: the GPU / mixed-precision code path (no GPU was available to the assistant) and execution on Google Colab.

## Incidents

- During clean-up, the assistant deleted `results/rnn/`, which contained a `model.keras` from a training run that was still in progress. The user had recreated the folder; its earlier contents were not recovered by the assistant.

## Reproducing the work

See "How to run" in `README.md`. The settings that define each run are stored in `results/<run>/<model>/run_config.json`.
