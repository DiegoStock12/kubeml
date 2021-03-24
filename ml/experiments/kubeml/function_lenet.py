""" Definition of a KubeML function to train the LeNet network with the MNIST dataset"""
import logging
import random
from typing import List, Any, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
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
        super().__init__(network, dataset)

    def init(self, model: nn.Module):
        pass

    def train(self, model: nn.Module, dataset: KubeDataset) -> float:

        # parse the kubernetes args
        batch = self.args.batch_size
        lr = self.args.lr

        # define the device for training and load the data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = data.DataLoader(dataset, batch_size=batch)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # get the outputs
            optimizer.zero_grad()
            output = model(x)

            # compute loss and backprop
            logging.debug(f'Shape of the output is {output.shape}, y is {y.shape}')
            loss = loss_fn(output, y)
            loss.backward()

            # step with the optimizer
            optimizer.step()
            total_loss += loss.item()

            ## TODO change the loss for a function that the user calls like tensorboard
            if batch_idx % 10 == 0:
                logging.info(f"Index {batch_idx}, error: {loss.item}")

        return total_loss / len(train_loader)

    def validate(self, model: nn.Module, dataset: KubeDataset) -> Tuple[float, float]:
        batch = self.args.batch_size

        # define the device for training and load the data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_loader = data.DataLoader(dataset, batch_size=batch)
        loss_fn = nn.CrossEntropyLoss()

        model.eval()

        test_loss = 0
        correct = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            test_loss += loss_fn(output, y).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()

        test_loss /= len(val_loader)
        accuracy = 100. * correct / len(val_loader.dataset)

        return accuracy, test_loss

    def infer(self, model: nn.Module, data: List[Any]) -> Union[torch.Tensor, np.ndarray, List[float]]:
        pass


def main():
    # set the random seeds
    torch.manual_seed(42)
    random.seed(42)

    lenet = LeNet()
    dataset = MnistDataset()
    kubenet = KubeLeNet(lenet, dataset)
    return kubenet.start()
