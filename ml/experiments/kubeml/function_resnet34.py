import logging
import random
from typing import List, Any, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from kubeml import KubeModel, KubeDataset
from torch.optim import SGD
from torchvision.models.resnet import resnet34


class Cifar10Dataset(KubeDataset):
    def __init__(self):
        super(Cifar10Dataset, self).__init__("cifar10")

        self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.train_transf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            self.normalize
        ])

        self.val_transf = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        # depending on the mode of the dataset apply some or the other
        # transformations
        if self.is_training():
            return self.train_transf(x), y.astype('int64')
        else:
            return self.val_transf(x), y.astype('int64')

    def __len__(self):
        return len(self.data)


class KubeResnet34(KubeModel):
    def __init__(self, network, dataset: Cifar10Dataset):
        super(KubeResnet34, self).__init__(network, dataset, gpu=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        sgd = SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        return sgd

    def train(self, batch, batch_index) -> float:
        criterion = nn.CrossEntropyLoss()

        # get the targets and labels from the batch
        x, y = batch

        self.optimizer.zero_grad()
        output = self(x)
        loss = criterion(output, y)

        loss.backward()
        self.optimizer.step()

        if batch_index % 10 == 0:
            logging.info(f"Index {batch_index}, error: {loss.item()}")

        return loss.item()

    def validate(self, batch, batch_index) -> Tuple[float, float]:
        # get the inputs
        x, y = batch

        criterion = nn.CrossEntropyLoss()

        output = self(x)
        _, predicted = torch.max(output.data, 1)
        test_loss = criterion(output, y).item()
        correct = predicted.eq(y).sum().item()

        accuracy = correct * 100 / self.batch_size

        return accuracy, test_loss


def main():
    # set the random seeds
    torch.manual_seed(42)
    random.seed(42)

    resnet = resnet34()
    dataset = Cifar10Dataset()
    kubenet = KubeResnet34(resnet, dataset)
    return kubenet.start()
