# Authors: Shadab, Aryan, Adina

from data import prepare_datasets
from models import build_cnn_model, build_rnn_model, build_simple_model
from training import train_and_plot


def main():
    dataset_train, dataset_test = prepare_datasets()

    model = build_simple_model()
    train_and_plot(model, dataset_train, dataset_test)

    RNN_model = build_rnn_model()
    RNN_model.summary()

    CNN_model = build_cnn_model()
    CNN_model.summary()


if __name__ == '__main__':
    main()
