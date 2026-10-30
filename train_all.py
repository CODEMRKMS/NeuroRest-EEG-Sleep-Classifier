# Authors: Shadab, Aryan, Adina

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # no display needed (Colab, remote machines)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

import config
import data
from models import BUILDERS

MODEL_ORDER = ('cnn', 'mlp', 'rnn')
MODEL_TITLES = {'mlp': 'MLP (SimpleNN)', 'cnn': '1D-CNN', 'rnn': 'SimpleRNN'}
DEFAULT_MAX_EPOCHS = {'mlp': 10, 'cnn': 30, 'rnn': 10}
V2_LEARNING_RATES = {'mlp': 1e-3, 'cnn': 1e-3, 'rnn': 1e-4}
REDUCE_LR = {'monitor': 'val_loss', 'factor': 0.5, 'patience': 2, 'min_lr': 1e-6}

RNN_TIMING_FRACTION = 0.05      # share of the training data used for the timing epoch
RNN_MAX_EPOCH_SECONDS = 300     # downsample if a projected full epoch is slower than this


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train the MLP, 1D-CNN and SimpleRNN on the subject-wise split')
    parser.add_argument('--model', choices=['mlp', 'cnn', 'rnn', 'all'], default='all',
                        help='model to train; "all" runs cnn, mlp, rnn in that order (default: %(default)s)')
    parser.add_argument('--data-dir', type=Path, default=config.DATA_DIR,
                        help='folder with the Sleep-EDF files and cache/ (default: %(default)s)')
    parser.add_argument('--results-dir', type=Path, default=config.RESULTS_DIR,
                        help='where results/<model>/ is written (default: %(default)s)')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='default: mlp %(mlp)d, cnn %(cnn)d, rnn %(rnn)d' % DEFAULT_MAX_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=128, help='same for all models (default: %(default)s)')
    parser.add_argument('--run-name', default='v2',
                        help='results go to <results-dir>/<run-name>/<model>/ (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Adam learning rate for every model being trained; default is the v2 value per model '
                             '(mlp %(mlp)g, cnn %(cnn)g, rnn %(rnn)g)' % V2_LEARNING_RATES)
    parser.add_argument('--patience', type=int, default=5,
                        help='EarlyStopping patience on val_loss (default: %(default)s)')
    parser.add_argument('--no-normalise', action='store_true',
                        help='use the raw EEG instead of per-recording, per-channel z-scores (v1 behaviour)')
    parser.add_argument('--no-reduce-lr', action='store_true',
                        help='disable ReduceLROnPlateau (v1 behaviour)')
    parser.add_argument('--seed', type=int, default=42, help='training seed (default: %(default)s)')
    parser.add_argument('--rnn-downsample', type=int, default=10,
                        help='average-pool factor along time for the RNN input if the full-length RNN is too '
                             'slow (3000 -> 3000/factor steps); 1 disables it (default: %(default)s)')
    parser.add_argument('--rnn-max-epoch-seconds', type=float, default=RNN_MAX_EPOCH_SECONDS,
                        help='projected full-epoch time above which the RNN input is downsampled '
                             '(default: %(default)s)')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                        help='random fraction of train and val epochs to use, for smoke tests (default: %(default)s)')
    return parser.parse_args(argv)


# ------------------------------------------------------------------- setup

def setup_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        name = tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'GPU')
        device = {'device': f'GPU ({name})', 'mixed_precision': True}
    else:
        device = {'device': 'CPU', 'mixed_precision': False}
    print(f'Device: {device["device"]} | mixed precision: {device["mixed_precision"]}')
    return device


def seed_everything(seed):
    tf.keras.utils.set_random_seed(seed)  # python, numpy and tensorflow


def make_dataset(X, y, batch_size, shuffle, seed):
    ds = tf.data.Dataset.from_tensor_slices((X, y)).cache()
    if shuffle:
        ds = ds.shuffle(min(len(X), config.SHUFFLE_BUFFER_SIZE), seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def downsample_time(X, factor):
    """Average-pool (N, T, C) along time to (N, T // factor, C); apply identically to every split."""
    n, t, c = X.shape
    if t % factor:
        raise ValueError(f'--rnn-downsample {factor} does not divide the {t} time steps')
    return X.reshape(n, t // factor, factor, c).mean(axis=2, dtype=np.float32)


def take_fraction(X, y, fraction, seed):
    """Random subset keeping the class proportions (and at least one epoch of every class)."""
    if fraction >= 1:
        return X, y
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        n = max(1, int(round(len(members) * fraction)))
        idx.append(rng.choice(members, n, replace=False))
    idx = np.sort(np.concatenate(idx))
    return X[idx], y[idx]


# --------------------------------------------------------------------- RNN

def decide_rnn_input(X_train, y_train, X_val, y_val, args, learning_rate):
    """Time one epoch on 5% of the data at full length and decide whether to downsample."""
    length = X_train.shape[1]
    Xs, ys = take_fraction(X_train, y_train, RNN_TIMING_FRACTION, args.seed)
    Xv, yv = take_fraction(X_val, y_val, RNN_TIMING_FRACTION, args.seed)
    fraction = len(Xs) / len(X_train)

    seed_everything(args.seed)
    model = BUILDERS['rnn']((length, X_train.shape[2]), learning_rate)
    train_ds = make_dataset(Xs, to_categorical(ys, 5), args.batch_size, False, args.seed)
    val_ds = make_dataset(Xv, to_categorical(yv, 5), args.batch_size, False, args.seed)
    model.fit(train_ds.take(1), epochs=1, shuffle=False, verbose=0)  # warm-up: exclude graph tracing from the timing
    start = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=1, shuffle=False, verbose=0)
    measured = time.perf_counter() - start
    projected = measured / fraction

    downsample = projected > args.rnn_max_epoch_seconds and args.rnn_downsample > 1
    factor = args.rnn_downsample if downsample else 1
    decision = {
        'original_length': length,
        'downsample_factor': factor,
        'new_length': length // factor,
        'downsampled': downsample,
        'timing_sample_fraction': round(fraction, 4),
        'timing_sample_epochs': len(Xs),
        'measured_sample_epoch_sec': round(measured, 2),
        'projected_full_epoch_sec': round(projected, 1),
        'threshold_sec': args.rnn_max_epoch_seconds,
    }
    verdict = 'DOWNSAMPLING' if downsample else 'keeping the full length'
    if projected > args.rnn_max_epoch_seconds and not downsample:
        verdict = 'keeping the full length (--rnn-downsample 1)'
    print(f'RNN input decision: original length {length}, measured {measured:.1f}s for {len(Xs)} epochs '
          f'({fraction:.1%}) -> projected {projected:.0f}s per full epoch '
          f'(threshold {args.rnn_max_epoch_seconds:.0f}s); {verdict}: '
          f'factor {factor}, new length {length // factor}')
    return decision


# ---------------------------------------------------------------- training

class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self._start
        self.times.append(elapsed)
        print(f'  epoch {epoch + 1} time: {elapsed:.1f}s')


def plot_curves(history, out_dir, title):
    for metric, label, filename in (('accuracy', 'Accuracy', 'accuracy_curve.png'),
                                    ('loss', 'Loss', 'loss_curve.png')):
        plt.figure(figsize=(6, 4))
        plt.plot(history['epoch'], history[metric], label='Train')
        plt.plot(history['epoch'], history[f'val_{metric}'], label='Validation')
        plt.title(f'{title}: {label.lower()}')
        plt.xlabel('Epoch')
        plt.ylabel(label)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=150)
        plt.close()


def train_model(name, X_train, y_train, X_val, y_val, args, learning_rate, run_info, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)  # each model starts from the same seed regardless of what ran before it
    max_epochs = args.max_epochs or DEFAULT_MAX_EPOCHS[name]
    input_shape = tuple(X_train.shape[1:])

    model = BUILDERS[name](input_shape, learning_rate)
    model.summary()

    weights = compute_class_weight('balanced', classes=np.arange(len(config.CLASS_NAMES)), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(weights)}
    print('Class weights: ' + ', '.join(f'{config.CLASS_NAMES[i]}={w:.3f}' for i, w in class_weight.items()))

    train_ds = make_dataset(X_train, to_categorical(y_train, 5), args.batch_size, True, args.seed)
    val_ds = make_dataset(X_val, to_categorical(y_val, 5), args.batch_size, False, args.seed)

    model_path = out_dir / 'model.keras'
    timer = EpochTimer()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=args.patience, restore_best_weights=True)
    callbacks = [timer, early_stopping]
    if not args.no_reduce_lr:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**REDUCE_LR))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        model_path, monitor='val_loss', save_best_only=True))  # best weights are on disk even if a later epoch crashes

    start = time.perf_counter()
    fit = model.fit(train_ds, validation_data=val_ds, epochs=max_epochs, class_weight=class_weight,
                    callbacks=callbacks, shuffle=False, verbose=2)
    total_time = time.perf_counter() - start
    if not model_path.exists():  # val_loss never improved (e.g. NaN)
        model.save(model_path)

    history = pd.DataFrame(fit.history)
    history.insert(0, 'epoch', np.arange(1, len(history) + 1))
    history['epoch_time_sec'] = np.round(timer.times, 3)
    columns = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy', 'epoch_time_sec']
    if 'learning_rate' in history:  # logged by ReduceLROnPlateau
        columns.append('learning_rate')
    history = history[columns]
    history.to_csv(out_dir / 'history.csv', index=False)
    plot_curves(history, out_dir, MODEL_TITLES[name])

    stopped_early = early_stopping.stopped_epoch > 0
    best_epoch = int(history['val_loss'].idxmin()) + 1
    run_config = {
        'run_name': args.run_name,
        'model': name,
        'architecture': MODEL_TITLES[name],
        'optimizer': 'Adam',
        'learning_rate': learning_rate,
        'final_learning_rate': float(history['learning_rate'].iloc[-1]) if 'learning_rate' in history
                               else learning_rate,
        'reduce_lr_on_plateau': {'enabled': not args.no_reduce_lr, **REDUCE_LR},
        'loss': 'categorical_crossentropy',
        'batch_size': args.batch_size,
        'max_epochs': max_epochs,
        'epochs_run': len(history),
        'early_stopping': {'monitor': 'val_loss', 'patience': args.patience, 'restore_best_weights': True,
                           'stopped_early': stopped_early,
                           'early_stop_epoch': early_stopping.stopped_epoch + 1 if stopped_early else None,
                           'best_epoch': best_epoch},
        'class_weights': {config.CLASS_NAMES[i]: w for i, w in class_weight.items()},
        'input_shape': list(input_shape),
        'n_train_epochs': len(X_train),
        'n_val_epochs': len(X_val),
        'data_fraction': args.data_fraction,
        'seed': args.seed,
        'total_train_time_sec': round(total_time, 1),
        'tensorflow_version': tf.__version__,
        'keras_version': tf.keras.__version__,
        'parameters': int(model.count_params()),
        **run_info,
    }
    with open(out_dir / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)

    print(f'{MODEL_TITLES[name]} finished: {len(history)} epochs in {total_time:.1f}s, saved to {out_dir}')
    return {'model': name, 'epochs': len(history),
            'best_val_accuracy': float(history['val_accuracy'].max()),
            'best_val_loss': float(history['val_loss'].min()),
            'time_sec': total_time}


def run_model(name, splits, args, run_info):
    out_dir = args.results_dir / args.run_name / name
    learning_rate = args.lr if args.lr is not None else V2_LEARNING_RATES[name]
    X_train, y_train = splits['train']['X'], splits['train']['y']
    X_val, y_val = splits['val']['X'], splits['val']['y']
    info = dict(run_info)

    if name == 'rnn':
        out_dir.mkdir(parents=True, exist_ok=True)
        decision = decide_rnn_input(X_train, y_train, X_val, y_val, args, learning_rate)
        info['rnn_downsampling'] = decision
        with open(out_dir / 'run_config.json', 'w') as f:  # kept even if training later fails
            json.dump({'model': name, 'status': 'input decision only', **info}, f, indent=2)
        if decision['downsampled']:
            X_train = downsample_time(X_train, decision['downsample_factor'])
            X_val = downsample_time(X_val, decision['downsample_factor'])
    else:
        info['rnn_downsampling'] = None

    return train_model(name, X_train, y_train, X_val, y_val, args, learning_rate, info, out_dir)


# -------------------------------------------------------------------- main

def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = setup_device()

    # The test signals are never loaded here; only its labels, for the split check below.
    splits = data.load_splits(args.data_dir, with_test_x=False, normalise=not args.no_normalise)
    for name in ('train', 'val', 'test'):
        data.print_split_summary(name, splits[name]['subjects'], splits[name]['y'])
    test_md5 = hashlib.md5(splits['test']['y'].astype(np.int8).tobytes()).hexdigest()
    print(f'Test labels MD5: {test_md5}')

    for i, name in enumerate(('train', 'val')):
        X, y = take_fraction(splits[name]['X'], splits[name]['y'], args.data_fraction, args.seed + i)
        splits[name]['X'], splits[name]['y'] = X.astype(np.float32, copy=False), y
    if args.data_fraction < 1:
        print(f'Using {args.data_fraction:.1%} of train and val: '
              f'{len(splits["train"]["y"])} / {len(splits["val"]["y"])} epochs')

    run_info = {
        **device,
        'split': {name: {'subjects': splits[name]['subjects'],
                         'class_counts': dict(zip(config.CLASS_NAMES,
                                                  np.bincount(splits[name]['y'], minlength=5).tolist()))}
                  for name in ('train', 'val', 'test')},
        'test_labels_md5': test_md5,
        'normalisation': {'enabled': not args.no_normalise,
                          'method': 'per-recording, per-channel z-score over the whole night (wake included)'
                                    if not args.no_normalise else None,
                          'cache_version': config.NORM_VERSION if not args.no_normalise else 'raw'},
        'data_dir': str(args.data_dir),
    }

    names = MODEL_ORDER if args.model == 'all' else (args.model,)
    summaries, failed = [], []
    for name in names:
        print(f'\n===== {MODEL_TITLES[name]} =====')
        try:
            summaries.append(run_model(name, splits, args, run_info))
        except Exception:
            traceback.print_exc()
            failed.append(name)

    print('\nSummary')
    print(f'{"model":<8}{"epochs":>8}{"best val_acc":>14}{"best val_loss":>15}{"total time":>13}')
    for s in summaries:
        print(f'{s["model"]:<8}{s["epochs"]:>8}{s["best_val_accuracy"]:>14.4f}'
              f'{s["best_val_loss"]:>15.4f}{s["time_sec"]:>12.1f}s')
    if failed:
        print(f'FAILED: {", ".join(failed)}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
