# Authors: Shadab, Aryan, Adina

import glob
import os

import numpy as np
import pyedflib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def prepare_datasets(path='/content/Samples'):
    # Getting all files together, separating hypnogram and PSG files, and re-ordering them.
    c = glob.glob(os.path.join(path, '*Hypnogram.edf'))
    d = glob.glob(os.path.join(path, '*PSG.edf'))
    len_annot = len(glob.glob(os.path.join(path, '*Hypnogram.edf')))
    len_signal = len(glob.glob(os.path.join(path, '*PSG.edf')))
    c.sort()
    d.sort()

    X_data_total = np.zeros((0, 3000, 2))
    Y_data_total = np.zeros(0)

    for i in range(0, len_annot):
        f = pyedflib.EdfReader(d[i])
        g = pyedflib.EdfReader(c[i])
        n = f.signals_in_file

        sigbufs = np.zeros((n - 5, f.getNSamples()[0]))
        for i in np.arange(n - 5):
            sigbufs[i, :] = f.readSignal(i)

        annots = g.readAnnotations()
        annots_norm = np.empty((len(annots) - 1, len(annots[0])))
        annots_norm[0:2, :] = np.asarray(annots[0:2], dtype=np.int64)
        annots_str = np.asarray(annots[2])

        annots_norm[0, :] = annots_norm[0, :] / 30
        annots_norm[1, :] = annots_norm[1, :] / 30

        annots_strtoclass_1 = np.char.replace(annots_str, ['Sleep stage W'], ['0'])
        annots_strtoclass_2 = np.char.replace(annots_strtoclass_1, ['Sleep stage 1'], ['1'])
        annots_strtoclass_3 = np.char.replace(annots_strtoclass_2, ['Sleep stage 2'], ['2'])
        annots_strtoclass_4 = np.char.replace(annots_strtoclass_3, ['Sleep stage 3'], ['3'])
        annots_strtoclass_5 = np.char.replace(annots_strtoclass_4, ['Sleep stage 4'], ['4'])
        annots_strtoclass_6 = np.char.replace(annots_strtoclass_5, ['Sleep stage R'], ['5'])
        annots_strtoclass_7 = np.char.replace(annots_strtoclass_6, ['Sleep stage ?'], ['6'])
        annots_strtoclass_8 = np.char.replace(annots_strtoclass_7, ['Movement time'], ['7'])
        annots_strtoclass = annots_strtoclass_8.astype(np.int64)

        count = 0
        k = 0
        X_data = np.zeros(shape=(int(sigbufs[0].size / 3000), 3000, 2))
        Y_data = np.zeros(int(sigbufs[0].size / 3000))

        for i in range(annots_norm[0].size - 1):
            for j in range(int(annots_norm[1, i])):
                X_data[k, :, :] = np.transpose(sigbufs[:, count:count + 3000])
                Y_data[k] = annots_strtoclass[i]
                k = k + 1
                count = count + 3000
        X_data_total = np.append(X_data_total, X_data, axis=0)
        Y_data_total = np.append(Y_data_total, Y_data)

    remove = np.where(Y_data_total == 7)
    remove = np.array(remove)
    X_data_total = np.delete(X_data_total, remove, axis=0)
    Y_data_total = np.delete(Y_data_total, remove)

    remove1 = np.where(Y_data_total == 0)
    remove1 = np.array(remove1)
    X_data_total = np.delete(X_data_total, remove1, axis=0)
    Y_data_total = np.delete(Y_data_total, remove1)

    sleep_set = X_data_total
    y1 = to_categorical(Y_data_total)
    y = y1[:, 1:6]

    training_data, testing_data, training_labels, testing_labels = train_test_split(
        sleep_set, y, test_size=0.2
    )

    dataset_train = tf.data.Dataset.from_tensor_slices((training_data, training_labels))
    dataset_test = tf.data.Dataset.from_tensor_slices((testing_data, testing_labels))

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 10000

    dataset_train = dataset_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    dataset_test = dataset_test.batch(BATCH_SIZE)

    return dataset_train, dataset_test
