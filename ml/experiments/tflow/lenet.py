import keras
from keras import layers
from keras.utils import np_utils
from keras.optimizers import SGD
import os

import tensorflow as tf

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


def main(epochs: int, batch: int):
    x_train, x_test, y_train, y_test = load_data()
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
    y_train, y_test = np_utils.to_categorical(y_train), np_utils.to_categorical(y_test)

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    mu, std = x_train.mean().astype(np.float32), x_train.mean().astype(np.float32)
    x_train -= mu
    x_test -= mu
    x_train /= std
    x_train /= std

    print(x_train.shape, x_test.shape)

    sgd = SGD(lr=0.1, momentum=0.9)

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
