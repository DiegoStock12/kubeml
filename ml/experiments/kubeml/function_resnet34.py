from typing import List, Any, Union, Tuple

import numpy as np
import torch
import logging

import torch.nn as nn
import torch.utils.data as data
from torch.optim import Adam
import torchvision.transforms as transforms
from kubeml import KubeModel, KubeDataset
from torchvision.models.resnet import resnet34


class Cifar10Dataset(KubeDataset):
    def __init__(self):
        super(Cifar10Dataset, self).__init__("cifar10")
        self.transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        return self.transf(x), y.astype('int64')

    def __len__(self):
        return len(self.data)


class KubeResnet34(KubeModel):
    def __init__(self, network, dataset: Cifar10Dataset):
        super(KubeResnet34, self).__init__(network, dataset)

    def init(self, model: nn.Module):
        pass

    def train(self, model: nn.Module, dataset: KubeDataset) -> float:

        loader = data.DataLoader(dataset, batch_size=self.args.batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=self.args.lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.train()
        total_loss = 0
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 10 == 0:
                logging.info(f"Index {i}, error: {loss.item}")

        return total_loss / len(loader)

    def validate(self, model: nn.Module, dataset: KubeDataset) -> Tuple[float, float]:

        loader = data.DataLoader(dataset, batch_size=self.args.batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()

        model.eval()
        total = 0
        correct = 0
        test_loss = 0
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)
            test_loss += criterion(output, labels).item()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = correct / total * 100
        test_loss /= len(loader)

        return accuracy, test_loss

    def infer(self, model: nn.Module, data: List[Any]) -> Union[torch.Tensor, np.ndarray, List[float]]:
        pass


def main():
    resnet = resnet34()
    dataset = Cifar10Dataset()
    kubenet = KubeResnet34(resnet, dataset)
    return kubenet.start()
