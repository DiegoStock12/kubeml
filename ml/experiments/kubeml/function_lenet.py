""" Definition of a KubeML function to train the LeNet network with the MNIST dataset"""
import logging
import random
from typing import List, Any, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from kubeml import KubeModel, KubeDataset
from torch.optim import SGD


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


class MnistDataset(KubeDataset):

    def __init__(self):
        super().__init__("mnist")
        self.transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        return self.transf(x), y.astype('int64')

    def __len__(self):
        return len(self.data)


class KubeLeNet(KubeModel):

    def __init__(self, network: nn.Module, dataset: MnistDataset):
        super().__init__(network, dataset, gpu=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        sgd = SGD(self._network.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        return sgd

    def init(self, model: nn.Module):
        pass

    def train(self, x, y, batch_index) -> float:
        # define the device for training and load the data
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0

        self.optimizer.zero_grad()
        output = self(x)

        # compute loss and backprop
        logging.debug(f'Shape of the output is {output.shape}, y is {y.shape}')
        loss = loss_fn(output, y)
        loss.backward()

        # step with the optimizer
        self.optimizer.step()
        total_loss += loss.item()

        if batch_index % 10 == 0:
            logging.info(f"Index {batch_index}, error: {loss.item()}")

        return total_loss

    def validate(self, x, y, _) -> Tuple[float, float]:
        loss_fn = nn.CrossEntropyLoss()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            output = self(x)
            test_loss += loss_fn(output, y).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()

        accuracy = 100. * correct
        return accuracy, test_loss

    def infer(self, data: List[Any]) -> Union[torch.Tensor, np.ndarray, List[float]]:
        pass


def main():
    # set the random seeds
    torch.manual_seed(42)
    random.seed(42)

    lenet = LeNet()
    dataset = MnistDataset()
    kubenet = KubeLeNet(lenet, dataset)
    return kubenet.start()
