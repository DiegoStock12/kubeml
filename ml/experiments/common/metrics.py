import matplotlib.pyplot as plt
import numpy as np

from .experiment import History


def plot_validation_loss(h: History):
    """Plots the validation loss against the number of epochs"""
    plt.figure(figsize=(10, 5))
    val_loss = h.data.validation_loss
    step = h.task.options.validate_every
    x = range(1, (len(val_loss) * step) + 1, step) if step != 0 else [1]

    plt.legend()
    plt.plot(x, val_loss, label='val loss')


def plot_accuracy(h: History):
    """Plots the accuracy against the number of epochs"""
    plt.figure(figsize=(10, 5))
    acc = h.data.accuracy
    step = h.task.options.validate_every
    x = range(1, (len(acc) * step) + 1, step) if step != 0 else [1]

    plt.legend()
    plt.plot(x, acc, label='val accuracy')


def plot_train_loss(h: History):
    """Plots the train loss against the number of epochs"""
    plt.figure(figsize=(10, 5))
    train_loss = h.data.validation_loss
    x = range(1, len(train_loss) + 1)

    plt.plot(x, train_loss, label='train loss')
