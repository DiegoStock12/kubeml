
import numpy as np
from typing import Tuple
import argparse
import os
import tensorflow as tf

import keras_resnet.models
import resnet_classes
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

CIFAR_LOCATION = "../datasets/cifar10"
DATASET = 'cifar10'


# load the cifar10 train and val data from storage
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.load(os.path.join(CIFAR_LOCATION, f'{DATASET}_x_train.npy'))
    x_val = np.load(os.path.join(CIFAR_LOCATION, f'{DATASET}_x_test.npy'))
    y_train = np.load(os.path.join(CIFAR_LOCATION, f'{DATASET}_y_train.npy'))
    y_test = np.load(os.path.join(CIFAR_LOCATION, f'{DATASET}_y_test.npy'))

    return x_train, x_val, y_train, y_test


def main(num_epochs: int, batch_size: int):
    shape = (32, 32, 3)
    n_classes = 10
    x = Input(shape)

    # load the data and preprocess
    x_train, x_test, y_train, y_test = load_data()
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
    y_train, y_test =to_categorical(y_train), to_categorical(y_test)

    # subtract mean and normalize
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image
    x_train /= 128.
    x_test /= 128.

    sgd = SGD(lr=0.1, momentum=0.9)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # build and compile model
    with strategy.scope():
        model = keras_resnet.models.ResNet34(x, classes=n_classes)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=int(batch_size),
                        epochs=int(num_epochs),
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
