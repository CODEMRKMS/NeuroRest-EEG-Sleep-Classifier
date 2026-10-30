# Authors: Shadab, Aryan, Adina

import argparse
from pathlib import Path

import config
from data import prepare_datasets
from models import build_cnn_model, build_rnn_model, build_simple_model
from training import train_and_plot


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='NeuroRest EEG sleep-stage classifier')
    parser.add_argument('--data-dir', type=Path, default=config.DATA_DIR,
                        help='folder with *PSG.edf / *Hypnogram.edf files (default: %(default)s)')
    parser.add_argument('--num-subjects', type=int, default=config.NUM_SUBJECTS,
                        help='subjects to download when the folder has no EDF files (default: %(default)s)')
    parser.add_argument('--download', action='store_true',
                        help='download any missing files even if the folder already has some EDFs')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset_train, dataset_val, dataset_test = prepare_datasets(
        args.data_dir, args.num_subjects, args.download)

    model = build_simple_model()
    train_and_plot(model, dataset_train, dataset_val, dataset_test)

    RNN_model = build_rnn_model()
    RNN_model.summary()

    CNN_model = build_cnn_model()
    CNN_model.summary()


if __name__ == '__main__':
    main()
