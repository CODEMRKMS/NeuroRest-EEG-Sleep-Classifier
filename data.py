# Authors: Shadab, Aryan, Adina

import os
import re
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyedflib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

import config

# Hypnogram description -> class index (S1=0 ... REM=4). Wake, '?' and Movement time
# are dropped, so only the 5 sleep classes remain.
STAGE_TO_CLASS = {
    'Sleep stage 1': 0,
    'Sleep stage 2': 1,
    'Sleep stage 3': 2,
    'Sleep stage 4': 3,
    'Sleep stage R': 4,
}
DROPPED_STAGES = {'Sleep stage W', 'Sleep stage ?', 'Movement time'}


# ---------------------------------------------------------------- download

def download_sleep_edf(data_dir, num_subjects=config.NUM_SUBJECTS, base_url=config.BASE_URL):
    """Download PSG + Hypnogram pairs for the first `num_subjects` subjects (both nights)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(base_url, timeout=60) as response:
        html = response.read().decode('utf-8')
    names = sorted(set(re.findall(r'href="(SC4\d{3}[A-Z0-9]{2}-(?:PSG|Hypnogram)\.edf)"', html)))

    subjects = sorted({name[3:5] for name in names})[:num_subjects]
    names = [name for name in names if name[3:5] in subjects]
    print(f'Downloading {len(names)} files for {len(subjects)} subjects to {data_dir}')

    for n, name in enumerate(names, 1):
        target = data_dir / name
        if target.exists():
            continue
        print(f'[{n}/{len(names)}] {name}')
        _download_file(base_url + name, target)


def _download_file(url, target, retries=3):
    part = target.with_name(target.name + '.part')
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(part, 'wb') as out:
                shutil.copyfileobj(response, out, length=1 << 20)
            os.replace(part, target)
            return
        except OSError as error:
            if attempt == retries:
                raise
            print(f'  retry {attempt}/{retries - 1} after: {error}')
            time.sleep(2 * attempt)


# ----------------------------------------------------------------- loading

def find_recordings(data_dir):
    """Pair PSG and Hypnogram files by their recording id (e.g. SC4001)."""
    data_dir = Path(data_dir)
    psg = {p.name[:6]: p for p in data_dir.glob('*PSG.edf')}
    hyp = {p.name[:6]: p for p in data_dir.glob('*Hypnogram.edf')}
    unpaired = sorted(psg.keys() ^ hyp.keys())
    if unpaired:
        print(f'Warning: ignoring recordings without a PSG/Hypnogram pair: {unpaired}')
    return [(key, psg[key], hyp[key]) for key in sorted(psg.keys() & hyp.keys())]


def read_recording(psg_path, hyp_path, normalise=True):
    """Cut one night into 30 s epochs of shape (3000, 2) with sleep-stage labels 0-4.

    With normalise=True each channel is z-scored with the mean/std of this night alone
    (whole recording, wake included), so nothing is shared between recordings or subjects.
    Returns X, y and the per-channel (mean, std) used (None, None if not normalised).
    """
    f = pyedflib.EdfReader(str(psg_path))
    g = pyedflib.EdfReader(str(hyp_path))
    try:
        labels = f.getSignalLabels()
        idx = [labels.index(channel) for channel in config.CHANNELS]  # Fpz-Cz, Pz-Oz
        for i in idx:
            if int(round(f.getSampleFrequency(i))) * config.EPOCH_SECONDS != config.SAMPLES_PER_EPOCH:
                raise ValueError(f'{psg_path.name}: expected 100 Hz EEG')
        sigbufs = np.stack([f.readSignal(i) for i in idx])
        onsets, durations, descriptions = g.readAnnotations()
    finally:
        f.close()
        g.close()

    mean = std = None
    if normalise:
        mean = sigbufs.mean(axis=1)
        std = np.maximum(sigbufs.std(axis=1), 1e-8)
        sigbufs = (sigbufs - mean[:, None]) / std[:, None]
    sigbufs = sigbufs.astype(np.float32)

    n_samples = sigbufs.shape[1]
    X, y = [], []
    for onset, duration, description in zip(onsets, durations, descriptions):
        if description in DROPPED_STAGES:
            continue
        if description not in STAGE_TO_CLASS:
            raise ValueError(f'{hyp_path.name}: unknown annotation {description!r}')
        start = int(round(float(onset) * 100))
        for j in range(int(float(duration) // config.EPOCH_SECONDS)):
            s = start + j * config.SAMPLES_PER_EPOCH
            if s + config.SAMPLES_PER_EPOCH > n_samples:
                break
            X.append(np.transpose(sigbufs[:, s:s + config.SAMPLES_PER_EPOCH]))
            y.append(STAGE_TO_CLASS[description])

    return np.stack(X), np.array(y, dtype=np.int8), mean, std


def cache_recordings(data_dir, normalise=True):
    """Return [(subject, cache_path)] per recording, reading EDFs only if there is no .npz cache.

    Normalised files are named <id>.<NORM_VERSION>.npz; the un-normalised ones keep <id>.npz,
    so one kind is never mistaken for the other.
    """
    data_dir = Path(data_dir)
    cache_dir = data_dir / 'cache'
    cache_dir.mkdir(exist_ok=True)
    suffix = f'.{config.NORM_VERSION}' if normalise else ''

    cached = []
    for key, psg_path, hyp_path in find_recordings(data_dir):
        cache = cache_dir / f'{key}{suffix}.npz'
        if not cache.exists():
            print(f'Processing {psg_path.name} -> {cache.name}')
            X, y, mean, std = read_recording(psg_path, hyp_path, normalise)
            tmp = cache_dir / f'{key}{suffix}.tmp.npz'
            if normalise:
                np.savez(tmp, X=X, y=y, mean=mean, std=std)
            else:
                np.savez(tmp, X=X, y=y)
            os.replace(tmp, cache)
        cached.append((key[3:5], cache))
    return cached


# ------------------------------------------------------------------- split

def split_subjects(subjects, seed=config.SEED,
                   val_fraction=config.VAL_FRACTION, test_fraction=config.TEST_FRACTION):
    """Shuffle the subject ids with a fixed seed and cut them into train/val/test."""
    subjects = sorted(set(subjects))
    n = len(subjects)
    if n < 3:
        raise ValueError(f'Need at least 3 subjects for a train/val/test split, found {n}')
    n_test = max(1, round(n * test_fraction))
    n_val = max(1, round(n * val_fraction))
    order = np.random.RandomState(seed).permutation(n)
    shuffled = [subjects[i] for i in order]
    return (sorted(shuffled[n_test + n_val:]),
            sorted(shuffled[n_test:n_test + n_val]),
            sorted(shuffled[:n_test]))


def print_split_summary(name, subjects, y):
    counts = np.bincount(y, minlength=len(config.CLASS_NAMES))
    print(f'{name}: {len(subjects)} subjects, {len(y)} epochs')
    print(f'  subjects: {", ".join(subjects)}')
    print('  classes:  ' + ', '.join(f'{c}={n}' for c, n in zip(config.CLASS_NAMES, counts)))


# ---------------------------------------------------------------- datasets

def load_splits(data_dir=config.DATA_DIR, num_subjects=config.NUM_SUBJECTS, download=False,
                with_test_x=True, normalise=True):
    """Subject-wise split as {'train'|'val'|'test': {'subjects', 'X', 'y'}}; y holds class ids 0-4.

    With with_test_x=False the test signals are never read from the cache (X is None),
    only its labels.
    """
    data_dir = Path(data_dir)

    if download or not find_recordings(data_dir):
        download_sleep_edf(data_dir, num_subjects)
    cached = cache_recordings(data_dir, normalise)
    if not cached:
        raise FileNotFoundError(f'No PSG/Hypnogram EDF pairs found in {data_dir}')

    train_ids, val_ids, test_ids = split_subjects([subject for subject, _ in cached])

    splits = {}
    for name, ids in (('train', train_ids), ('val', val_ids), ('test', test_ids)):
        Xs, ys = [], []
        for subject, path in cached:
            if subject not in ids:
                continue
            with np.load(path) as npz:  # .npz members are read lazily
                ys.append(npz['y'])
                if name != 'test' or with_test_x:
                    Xs.append(npz['X'])
        splits[name] = {'subjects': ids,
                        'X': np.concatenate(Xs) if Xs else None,
                        'y': np.concatenate(ys)}
    return splits


def prepare_datasets(data_dir=config.DATA_DIR, num_subjects=config.NUM_SUBJECTS, download=False):
    splits = load_splits(data_dir, num_subjects, download)

    datasets = []
    for name in ('train', 'val', 'test'):
        split = splits[name]
        print_split_summary(name, split['subjects'], split['y'])
        datasets.append((split['X'], to_categorical(split['y'], num_classes=len(config.CLASS_NAMES))))

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets
    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    dataset_train = dataset_train.shuffle(config.SHUFFLE_BUFFER_SIZE).batch(config.BATCH_SIZE)
    dataset_val = dataset_val.batch(config.BATCH_SIZE)
    dataset_test = dataset_test.batch(config.BATCH_SIZE)

    return dataset_train, dataset_val, dataset_test
