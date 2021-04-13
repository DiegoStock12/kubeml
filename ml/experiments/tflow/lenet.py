import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import History as KerasHistory
import os



from typing import Tuple
import numpy as np

import argparse


def get_model():
    input_shape = (28, 28, 1)
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(units=120, activation='relu'))
    model.add(layers.Dense(units=84, activation='relu'))
    model.add(layers.Dense(units=10, activation='softmax'))

    return model


MNIST_LOCATION = "../datasets/mnist"
DATASET = 'mnist'


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.load(os.path.join(MNIST_LOCATION, f'{DATASET}_x_train.npy'))
    x_val = np.load(os.path.join(MNIST_LOCATION, f'{DATASET}_x_test.npy'))
    y_train = np.load(os.path.join(MNIST_LOCATION, f'{DATASET}_y_train.npy'))
    y_test = np.load(os.path.join(MNIST_LOCATION, f'{DATASET}_y_test.npy'))

    return x_train, x_val, y_train, y_test


def main(epochs: int, batch: int) -> KerasHistory:
    x_train, x_test, y_train, y_test = load_data()
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image
    x_train /= 128.
    x_test /= 128.

    print(x_train.shape, x_test.shape)

    sgd = SGD(lr=0.01, momentum=0.9)

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = get_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
    # print(model.summary())

    history = model.fit(x_train, y_train,
                        batch_size=int(batch),
                        epochs=int(epochs),
                        validation_data=(x_test, y_test),
                        shuffle=True)

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help='number of epochs')
    parser.add_argument('-b', '--batch', help='batch size')
    args = parser.parse_args()

    print(args.batch, args.epochs)
    if args.batch is None or args.epochs is None:
        print("error: not clarified batch or epochs")
        exit(-1)

    h = main(args.epochs, args.batch)
