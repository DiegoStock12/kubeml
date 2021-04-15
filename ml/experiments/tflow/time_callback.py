import time

import tensorflow.keras as keras


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.start)
