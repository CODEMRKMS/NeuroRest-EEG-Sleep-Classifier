# Authors: Shadab, Aryan, Adina

import tensorflow as tf


def build_simple_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3000, 2)),
        tf.keras.layers.Dense(1500, activation='relu'),
        # tf.keras.layers.Dense(750, activation='relu'),
        # tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        # tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_rnn_model():
    RNN_model = tf.keras.Sequential(
        [
            # This is for an LSTM
            # tf.keras.layers.Embedding(input_dim=3000, output_dim=64),
            # tf.keras.layers.LSTM(128, return_sequences=False, recurrent_dropout=0.1, input_shape=(None,2)),
            # tf.keras.layers.LSTM(64, dropout=0.1),

            # This is for a GRU
            # tf.keras.layers.GRU(128, return_sequences=False, recurrent_dropout=0.2, input_shape=(None,2)),
            # tf.keras.layers.GRU(64, dropout=0.1),

            # This is for SimpleRNN
            tf.keras.layers.SimpleRNN(256, return_sequences=False, recurrent_dropout=0.2, input_shape=(3000, 2)),
            tf.keras.layers.Dense(1500, activation='relu'),
            # tf.keras.layers.Dense(750, activation='relu'),
            # tf.keras.layers.Dense(300, activation='relu'),
            # tf.keras.layers.Dense(50, activation='relu'),
            # tf.keras.layers.Dense(25, activation='relu'),
            # tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(5, activation='softmax'),
        ])

    RNN_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.000001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return RNN_model


def build_cnn_model():
    CNN_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(3000, 2)),
            # tf.keras.layers.Reshape(input_shape=(2,3000), target_shape=(2,3000,1)),

            tf.keras.layers.Conv1D(kernel_size=10, filters=50, activation='relu', padding='same', strides=2),
            tf.keras.layers.BatchNormalization(center=True, scale=False),
            tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),
            tf.keras.layers.Dropout(0.20),

            tf.keras.layers.Conv1D(kernel_size=10, filters=100, activation='relu', padding='same', strides=2),
            tf.keras.layers.BatchNormalization(center=True, scale=False),
            tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),
            tf.keras.layers.Dropout(0.20),

            # tf.keras.layers.Conv1D(kernel_size=10, filters=200, activation='relu', padding='same', strides=2),
            # tf.keras.layers.BatchNormalization(center=True, scale=False),
            # tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),
            # tf.keras.layers.Dropout(0.20),

            # tf.keras.layers.Conv1D(kernel_size=10, filters=400, activation='relu', padding='same', strides=2),
            # tf.keras.layers.BatchNormalization(center=True, scale=False),
            # tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),
            # tf.keras.layers.Dropout(0.20),

            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(1500, activation='relu'),
            # tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(5, activation='softmax')
        ])

    CNN_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return CNN_model
