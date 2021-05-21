"""code to train the vgg16 network"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import History as KerasHistory
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os

from time_callback import TimeHistory

from typing import Tuple

CIFAR_LOCATION = os.path.abspath(os.path.dirname(__file__)) + "/../datasets/cifar10"
DATASET = 'cifar10'


# load the cifar10 train and val data from storage
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.load(os.path.join(CIFAR_LOCATION, f'{DATASET}_x_train.npy'))
    x_val = np.load(os.path.join(CIFAR_LOCATION, f'{DATASET}_x_test.npy'))
    y_train = np.load(os.path.join(CIFAR_LOCATION, f'{DATASET}_y_train.npy'))
    y_test = np.load(os.path.join(CIFAR_LOCATION, f'{DATASET}_y_test.npy'))

    return x_train, x_val, y_train, y_test


def vgg16():
    # create the feature reduction layer
    model = Sequential()
    model.add(Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # add the classifier
    model.add(tf.keras.layers.Lambda(
        lambda image: tf.image.resize(
            image,
            (7, 7),
            method=tf.image.ResizeMethod.BICUBIC,
            # align_corners = True, # possibly important
            # preserve_aspect_ratio = True
        )
    ))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=10, activation="softmax"))

    return model


def main(num_epochs: int, batch_size: int) -> KerasHistory:
    shape = (32, 32, 3)
    n_classes = 10
    x = Input(shape)

    # load the data and preprocess
    x_train, x_test, y_train, y_test = load_data()
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    # subtract mean and normalize
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        # horizontal_flip=True
    )

    datagen.fit(x_train)

    train_iter = datagen.flow(x_train, y_train, batch_size=int(batch_size))
    test_iter = datagen.flow(x_test, y_test, batch_size=int(batch_size))



    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # build and compile model
    time_callback = TimeHistory()
    with strategy.scope():
        sgd = SGD(lr=0.01, momentum=0.9)
        model = vgg16()
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        print('setting weight decay')
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                layer.add_loss(lambda: tf.keras.regularizers.l2(5e-4)(layer.kernel))

    history = model.fit(train_iter,
                        batch_size=int(batch_size),
                        epochs=int(num_epochs),
                        validation_data=test_iter,
                        shuffle=True,
                        callbacks=[time_callback])

    return history, time_callback.times
