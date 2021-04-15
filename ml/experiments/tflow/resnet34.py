import numpy as np
from typing import Tuple
import argparse
import os
import tensorflow as tf

import keras_resnet.models

import tensorflow.keras as keras
from tensorflow.keras.callbacks import History as KerasHistory
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

from .time_callback import TimeHistory

CIFAR_LOCATION = os.path.abspath(os.path.dirname(__file__)) + "/../datasets/cifar10"
DATASET = 'cifar10'


# load the cifar10 train and val data from storage
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.load(os.path.join(CIFAR_LOCATION, f'{DATASET}_x_train.npy'))
    x_val = np.load(os.path.join(CIFAR_LOCATION, f'{DATASET}_x_test.npy'))
    y_train = np.load(os.path.join(CIFAR_LOCATION, f'{DATASET}_y_train.npy'))
    y_test = np.load(os.path.join(CIFAR_LOCATION, f'{DATASET}_y_test.npy'))

    return x_train, x_val, y_train, y_test


def main(num_epochs: int, batch_size: int) -> KerasHistory:
    shape = (32, 32, 3)
    n_classes = 10
    x = Input(shape)

    # load the data and preprocess
    x_train, x_test, y_train, y_test = load_data()
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    # subtract mean and normalize

    x_train /= 255.
    x_test /= 255.

    sgd = SGD(lr=0.1, momentum=0.9)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # build and compile model
    time_callback = TimeHistory()
    with strategy.scope():
        model = keras_resnet.models.ResNet34(x, classes=n_classes)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        print('setting weight decay')
        for layer in model.layers:
            if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
                layer.add_loss(lambda: keras.regularizers.l2(1e-4)(layer.kernel))

    history = model.fit(x_train, y_train,
                        batch_size=int(batch_size),
                        epochs=int(num_epochs),
                        validation_data=(x_test, y_test),
                        shuffle=True,
                        callbacks=[time_callback])

    return history, time_callback.times


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
