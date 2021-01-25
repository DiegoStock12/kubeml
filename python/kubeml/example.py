import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from .function import Network

net = nn.Module()


def train(loader: data.DataLoader, optim: optim.Optimizer):
    return {}


def validate():
    pass


def infer():
    pass


def init(m: nn.Module):
    pass


model = Network(
    model=net,
    init=init,
    train=train,
    inference=infer,
    validation=validate

)


def main():
    model.start()
