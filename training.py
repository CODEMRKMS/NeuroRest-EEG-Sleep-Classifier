# Authors: Shadab, Aryan, Adina

import matplotlib.pyplot as plt


def train_and_plot(model, dataset_train, dataset_test):
    model.summary()
    history = model.fit(dataset_train, epochs=10, validation_data=dataset_test)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig('Model Loss Comparison with 10 sets of Data, varying dropout.pdf')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig('Model Loss Comparison with 10 sets of Data, varying dropout.pdf')

    return history
