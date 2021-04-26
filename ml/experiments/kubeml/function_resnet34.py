import logging
import random
from typing import List, Any, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from kubeml import KubeModel, KubeDataset
from torch.optim import SGD
import torch.nn.functional as F
from torchvision.models.resnet import resnet34


class Cifar10Dataset(KubeDataset):
    def __init__(self):
        super(Cifar10Dataset, self).__init__("cifar10")

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
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
        # according to the original paper we should divide the lr by 10 after 32k iterations
        # and also after 48k iterations, stopping at 48k
        #
        # with a batch of 128 -> 390 iterations per epoch
        # that means we approximately divide after 80 and 120 epochs, and finish at 160
        if self.epoch > 80:
            self.lr *= 0.1
        elif self.epoch > 120:
            self.lr *= 0.01

        sgd = SGD(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return sgd

    def train(self, batch, batch_index) -> float:

        # get the targets and labels from the batch
        x, y = batch

        self.optimizer.zero_grad()
        output = self(x)
        loss = F.cross_entropy(output, y)

        loss.backward()
        self.optimizer.step()

        if batch_index % 10 == 0:
            logging.info(f"Index {batch_index}, error: {loss.item()}")

        return loss.item()

    def validate(self, batch, batch_index) -> Tuple[float, float]:
        # get the inputs
        x, y = batch

        output = self(x)
        _, predicted = torch.max(output.data, 1)
        test_loss = F.cross_entropy(output, y).item()
        correct = predicted.eq(y).sum().item()

        accuracy = correct * 100 / self.batch_size

        return accuracy, test_loss


def main():
    # set the random seeds
    # torch.manual_seed(42)
    # random.seed(42)

    resnet = resnet34()
    dataset = Cifar10Dataset()
    kubenet = KubeResnet34(resnet, dataset)
    return kubenet.start()
