# Authors: Shadab, Aryan, Adina

import tensorflow as tf

# Adam learning rates of the interim (v1) runs; train_all.py passes its own for v2.
LEARNING_RATES = {'mlp': 1e-3, 'rnn': 1e-6, 'cnn': 1e-5}


def build_simple_model(input_shape=(3000, 2), learning_rate=LEARNING_RATES['mlp']):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(1500, activation='relu'),
        # tf.keras.layers.Dense(750, activation='relu'),
        # tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        # tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5, activation='softmax', dtype='float32')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_rnn_model(input_shape=(3000, 2), learning_rate=LEARNING_RATES['rnn']):
    RNN_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),

            # This is for an LSTM
            # tf.keras.layers.Embedding(input_dim=3000, output_dim=64),
            # tf.keras.layers.LSTM(128, return_sequences=False, recurrent_dropout=0.1),
            # tf.keras.layers.LSTM(64, dropout=0.1),

            # This is for a GRU
            # tf.keras.layers.GRU(128, return_sequences=False, recurrent_dropout=0.2),
            # tf.keras.layers.GRU(64, dropout=0.1),

            # This is for SimpleRNN
            tf.keras.layers.SimpleRNN(256, return_sequences=False, recurrent_dropout=0.2),
            tf.keras.layers.Dense(1500, activation='relu'),
            # tf.keras.layers.Dense(750, activation='relu'),
            # tf.keras.layers.Dense(300, activation='relu'),
            # tf.keras.layers.Dense(50, activation='relu'),
            # tf.keras.layers.Dense(25, activation='relu'),
            # tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(5, activation='softmax', dtype='float32'),
        ])

    RNN_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return RNN_model


def build_cnn_model(input_shape=(3000, 2), learning_rate=LEARNING_RATES['cnn']):
    CNN_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
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
            tf.keras.layers.Dense(5, activation='softmax', dtype='float32')
        ])

    CNN_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return CNN_model


BUILDERS = {'mlp': build_simple_model, 'rnn': build_rnn_model, 'cnn': build_cnn_model}
