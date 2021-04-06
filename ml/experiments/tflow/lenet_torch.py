import random
from typing import List, Any, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.optim import SGD
import logging
import os


class LeNet(nn.Module):
    """ Definition of the LeNet network as per the 1998 paper

    Credits to https://github.com/ChawDoe/LeNet5-MNIST-PyTorch for the
    convenience of the network definition and the train loop found there
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


def train(model, loader):
    # define the device for training and load the data
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.cuda(), y.cuda()

        # get the outputs
        optimizer.zero_grad()
        output = model(x)

        # compute loss and backprop
        print(f'Shape of the output is {output.shape}, y is {y.shape}')
        loss = loss_fn(output, y)
        loss.backward()

        # step with the optimizer
        optimizer.step()
        total_loss += loss.item()

        ## TODO change the loss for a function that the user calls like tensorboard
        if batch_idx % 10 == 0:
            print(f"Index {batch_idx}, error: {loss.item}")


MNIST_LOCATION = "../datasets/mnist"
DATASET = 'mnist'


class MnistDataset(data.Dataset):
    def __init__(self):
        self.data, _, self.targets, _ = load_data()
        self.transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __getitem__(self, item):
        return self.transf(self.data[item]), self.targets[item].astype(np.int64)

    def __len__(self):
        return len(self.data)


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.load(os.path.join(MNIST_LOCATION, f'{DATASET}_x_train.npy'))
    x_val = np.load(os.path.join(MNIST_LOCATION, f'{DATASET}_x_test.npy'))
    y_train = np.load(os.path.join(MNIST_LOCATION, f'{DATASET}_y_train.npy'))
    y_test = np.load(os.path.join(MNIST_LOCATION, f'{DATASET}_y_test.npy'))

    return x_train, x_val, y_train, y_test


if __name__ == '__main__':
    dataset = MnistDataset()
    model = LeNet().cuda()
    optimizer = SGD(model.parameters(), momentum=0.9, lr=0.01)

    loader = data.DataLoader(dataset, batch_size=64)

    for i in range(5):
        print('Epoch', i)
        train(model, loader)
