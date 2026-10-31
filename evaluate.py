# Authors: Shadab, Aryan, Adina

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

import config
import data
from train_all import MODEL_TITLES, downsample_time

MODELS = ('cnn', 'rnn', 'mlp')
RUNS = ('v2', 'v1')  # v2 = main comparison, v1 = ablation
CLASSES = list(config.CLASS_NAMES)
COLORS = {'cnn': '#0072B2', 'rnn': '#E69F00', 'mlp': '#009E73'}  # colour-blind safe


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate the trained runs on the held-out test subjects')
    parser.add_argument('--data-dir', type=Path, default=config.DATA_DIR,
                        help='Sleep-EDF folder with cache/ (default: %(default)s)')
    parser.add_argument('--results-dir', type=Path, default=config.RESULTS_DIR,
                        help='folder holding <run>/<model>/; tables and plots are written here (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=128)
    return parser.parse_args(argv)


# --------------------------------------------------------------- test data

class TestData:
    """Test split of the subject-wise split, loaded once per normalisation setting."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._cache = {}
        self.subjects = None

    def get(self, normalise):
        if normalise not in self._cache:
            cached = data.cache_recordings(self.data_dir, normalise)
            _, _, test_ids = data.split_subjects([subject for subject, _ in cached])
            Xs, ys = [], []
            for subject, path in cached:
                if subject in test_ids:
                    with np.load(path) as npz:
                        Xs.append(npz['X'])
                        ys.append(npz['y'])
            self.subjects = test_ids
            self._cache[normalise] = (np.concatenate(Xs), np.concatenate(ys))
        return self._cache[normalise]


def preprocessing_from_config(cfg):
    """(normalise, rnn downsample factor) exactly as the run was trained."""
    normalise = bool(cfg.get('normalisation', {}).get('enabled', False))  # v1 configs predate this key: raw EEG
    downsampling = cfg.get('rnn_downsampling') or {}
    factor = int(downsampling.get('downsample_factor', 1)) if downsampling.get('downsampled') else 1
    return normalise, factor


# ------------------------------------------------------------- evaluation

def evaluate_run(run, model_name, args, test_data):
    run_dir = args.results_dir / run / model_name
    config_path, model_path = run_dir / 'run_config.json', run_dir / 'model.keras'
    if not config_path.exists() or not model_path.exists():
        return None, f'{run}/{model_name}: no run_config.json / model.keras in {run_dir}'
    cfg = json.load(open(config_path))
    if cfg.get('status'):
        return None, f'{run}/{model_name}: run_config.json status is {cfg["status"]!r} (run not finished)'

    normalise, factor = preprocessing_from_config(cfg)
    X, y = test_data.get(normalise)
    md5 = hashlib.md5(y.astype(np.int8).tobytes()).hexdigest()
    if md5 != cfg['test_labels_md5']:
        raise ValueError(f'{run}/{model_name}: test-label MD5 {md5} != run_config {cfg["test_labels_md5"]}')
    if factor > 1:
        X = downsample_time(X, factor)
    if list(X.shape[1:]) != list(cfg['input_shape']):
        raise ValueError(f'{run}/{model_name}: input shape {list(X.shape[1:])} != run_config {cfg["input_shape"]}')

    model = tf.keras.models.load_model(model_path, compile=False)
    pred = np.argmax(model.predict(X, batch_size=args.batch_size, verbose=0), axis=1)

    labels = np.arange(len(CLASSES))
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=labels, average='macro', zero_division=0)
    result = {
        'run': run, 'model': model_name, 'n_test': int(len(y)), 'normalise': normalise, 'downsample': factor,
        'accuracy': accuracy_score(y, pred),
        'macro_precision': precision, 'macro_recall': recall, 'macro_f1': f1,
        'cohen_kappa': cohen_kappa_score(y, pred),
        'per_class_f1': f1_score(y, pred, labels=labels, average=None, zero_division=0),
        'confusion': confusion_matrix(y, pred, labels=labels),
        'support': np.bincount(y, minlength=len(CLASSES)),
        'cfg': cfg,
        'history': pd.read_csv(run_dir / 'history.csv'),
    }
    print(f'{run}/{model_name}: normalise={normalise} downsample={factor} md5 ok | '
          f'acc={result["accuracy"]:.4f} macro_f1={result["macro_f1"]:.4f} kappa={result["cohen_kappa"]:.4f}')
    return result, None


# ------------------------------------------------------------------ tables

def metrics_row(r):
    row = {'run': r['run'], 'model': r['model'], 'n_test': r['n_test'],
           'accuracy': r['accuracy'], 'macro_precision': r['macro_precision'],
           'macro_recall': r['macro_recall'], 'macro_f1': r['macro_f1'], 'cohen_kappa': r['cohen_kappa']}
    row.update({f'f1_{c}': f for c, f in zip(CLASSES, r['per_class_f1'])})
    return row


def md_table(df):
    header = '| ' + ' | '.join(df.columns) + ' |'
    rule = '|' + '|'.join(['---'] * len(df.columns)) + '|'
    rows = ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join([header, rule, *rows])


def fmt(df):
    return df.round(4).astype(str) if len(df) else df


def build_tables(results, skipped, test_subjects, out_dir):
    rows = pd.DataFrame([metrics_row(r) for r in results])
    rows.round(4).to_csv(out_dir / 'comparison_table.csv', index=False)

    main = rows[rows.run == 'v2'].drop(columns=['run'])
    main.insert(0, 'Model', main.pop('model').map(MODEL_TITLES))
    parts = [f'## Main comparison (v2) - held-out test subjects {", ".join(test_subjects)}', '', md_table(fmt(main)), '']

    ablation = []
    for model in MODELS:
        v1 = rows[(rows.run == 'v1') & (rows.model == model)]
        v2 = rows[(rows.run == 'v2') & (rows.model == model)]
        if len(v1) and len(v2):
            a, b = v1.iloc[0], v2.iloc[0]
            row = {'Model': MODEL_TITLES[model]}
            for key, label in (('accuracy', 'accuracy'), ('macro_f1', 'macro F1'), ('cohen_kappa', 'kappa')):
                row[f'v1 {label}'], row[f'v2 {label}'] = a[key], b[key]
                row[f'delta {label}'] = b[key] - a[key]
            ablation.append(row)
    parts += ['## Ablation: v1 vs v2', '']
    parts += [md_table(fmt(pd.DataFrame(ablation))) if ablation else 'No model has both a v1 and a v2 run.', '']
    if skipped:
        parts += ['## Not evaluated', ''] + [f'- {s}' for s in skipped] + ['']
    (out_dir / 'comparison_table.md').write_text('\n'.join(parts), encoding='utf-8')


# ------------------------------------------------------------------- plots

def plot_confusion(r, out_path):
    cm = r['confusion'] / r['confusion'].sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f'{cm[i, j] * 100:.1f}%', ha='center', va='center',
                    color='white' if cm[i, j] > 0.5 else 'black', fontsize=9)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel('Predicted stage')
    ax.set_ylabel('True stage')
    ax.set_title(f'{MODEL_TITLES[r["model"]]} (v2)\nNormalised confusion matrix, test subjects')
    ax.title.set_fontsize(10)
    fig.colorbar(image, ax=ax, label='Fraction of true class')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(v2_results, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(v2_results)
    for k, r in enumerate(v2_results):
        x = np.arange(len(CLASSES)) + (k - (len(v2_results) - 1) / 2) * width
        ax.bar(x, r['per_class_f1'], width, label=MODEL_TITLES[r['model']], color=COLORS[r['model']])
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Sleep stage')
    ax.set_ylabel('F1 score')
    ax.set_title('Per-class F1 on held-out test subjects (v2)')
    ax.legend(loc='best')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ error notes

def top_confusions(r, k=5):
    cm = r['confusion']
    pairs = [(i, j, cm[i, j], cm[i, j] / cm[i].sum()) for i in range(len(CLASSES))
             for j in range(len(CLASSES)) if i != j and cm[i].sum()]
    return sorted(pairs, key=lambda p: p[3], reverse=True)[:k]


def history_gap(r):
    h = r['history']
    best = h.loc[h['val_loss'].idxmin()]
    last = h.iloc[-1]
    return best, last


def write_error_notes(results, skipped, test_subjects, md5, out_path):
    by_key = {(r['run'], r['model']): r for r in results}
    lines = ['# Error analysis notes (facts only)', '',
             f'- Test subjects: {", ".join(test_subjects)}; test-label MD5: {md5}',
             '- Train metrics in history.csv are Keras per-epoch running averages (dropout active); '
             'val and test metrics are computed on the full set in inference mode.', '',
             '## Class support (test)', '', '| Class | Epochs | Share |', '|---|---|---|']
    support = results[0]['support']
    lines += [f'| {c} | {n} | {n / support.sum() * 100:.1f}% |' for c, n in zip(CLASSES, support)]

    lines += ['', '## Top confused class pairs (true -> predicted, % of the true class)', '']
    for r in results:
        lines.append(f'**{r["run"]} {MODEL_TITLES[r["model"]]}**')
        lines += [f'- {CLASSES[i]} -> {CLASSES[j]}: {pct * 100:.1f}% ({n}/{r["support"][i]})'
                  for i, j, n, pct in top_confusions(r)]
        lines.append('')

    lines += ['## Train vs validation gap', '',
              '| Run | Model | Epochs run | Best epoch (val_loss) | Train acc | Val acc | Acc gap | Train loss | Val loss '
              '| Loss gap | Test acc | Val-test acc gap | Last-epoch acc gap |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        best, last = history_gap(r)
        lines.append(
            f'| {r["run"]} | {MODEL_TITLES[r["model"]]} | {len(r["history"])} | {int(best["epoch"])} '
            f'| {best["accuracy"]:.4f} | {best["val_accuracy"]:.4f} | {best["accuracy"] - best["val_accuracy"]:+.4f} '
            f'| {best["loss"]:.4f} | {best["val_loss"]:.4f} | {best["val_loss"] - best["loss"]:+.4f} '
            f'| {r["accuracy"]:.4f} | {best["val_accuracy"] - r["accuracy"]:+.4f} '
            f'| {last["accuracy"] - last["val_accuracy"]:+.4f} |')

    lines += ['', '## v1 vs v2 differences', '']
    compared = False
    for model in MODELS:
        a, b = by_key.get(('v1', model)), by_key.get(('v2', model))
        if not (a and b):
            continue
        compared = True
        ca, cb = a['cfg'], b['cfg']
        lines += [f'**{MODEL_TITLES[model]}**', '',
                  '| Item | v1 | v2 | v2 - v1 |', '|---|---|---|---|']
        for key, label in (('accuracy', 'Test accuracy'), ('macro_precision', 'Macro precision'),
                           ('macro_recall', 'Macro recall'), ('macro_f1', 'Macro F1'), ('cohen_kappa', "Cohen's kappa")):
            lines.append(f'| {label} | {a[key]:.4f} | {b[key]:.4f} | {b[key] - a[key]:+.4f} |')
        for c, fa, fb in zip(CLASSES, a['per_class_f1'], b['per_class_f1']):
            lines.append(f'| F1 {c} | {fa:.4f} | {fb:.4f} | {fb - fa:+.4f} |')
        lines += [
            f'| Learning rate (initial) | {ca["learning_rate"]} | {cb["learning_rate"]} | |',
            f'| Normalisation | {a["normalise"]} | {b["normalise"]} | |',
            f'| ReduceLROnPlateau | {ca.get("reduce_lr_on_plateau", {}).get("enabled", False)} '
            f'| {cb.get("reduce_lr_on_plateau", {}).get("enabled", False)} | |',
            f'| Early-stopping patience | {ca["early_stopping"]["patience"]} | {cb["early_stopping"]["patience"]} | |',
            f'| RNN downsample factor | {a["downsample"]} | {b["downsample"]} | |',
            f'| Epochs run | {ca["epochs_run"]} | {cb["epochs_run"]} | {cb["epochs_run"] - ca["epochs_run"]:+d} |',
            f'| Total train time (s) | {ca["total_train_time_sec"]} | {cb["total_train_time_sec"]} | |', '']
    if not compared:
        lines += ['No model has both a v1 and a v2 run.', '']
    if skipped:
        lines += ['## Not evaluated', ''] + [f'- {s}' for s in skipped] + ['']
    out_path.write_text('\n'.join(lines), encoding='utf-8')


# -------------------------------------------------------------------- main

def main(argv=None):
    args = parse_args(argv)
    test_data = TestData(args.data_dir)

    results, skipped = [], []
    for run in RUNS:
        for model in MODELS:
            result, reason = evaluate_run(run, model, args, test_data)
            if result:
                results.append(result)
            else:
                skipped.append(reason)
                print(f'skipped: {reason}')
    if not results:
        raise SystemExit('Nothing to evaluate')

    test_subjects = test_data.subjects
    md5 = results[0]['cfg']['test_labels_md5']
    build_tables(results, skipped, test_subjects, args.results_dir)
    v2 = [r for r in results if r['run'] == 'v2']
    for r in v2:
        plot_confusion(r, args.results_dir / f'confusion_matrix_v2_{r["model"]}.png')
    if v2:
        plot_per_class_f1(v2, args.results_dir / 'per_class_f1_v2.png')
    write_error_notes(results, skipped, test_subjects, md5, args.results_dir / 'error_analysis_notes.md')
    print(f'Wrote comparison_table.csv/.md, error_analysis_notes.md and plots to {args.results_dir}')


if __name__ == '__main__':
    main()
