from typing import Callable, Dict, List, Optional, Tuple, Any

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from .dataset import KubeArgs


class Model:

    def __init__(self,
                 network: nn.Module,
                 init: Optional[Callable[[nn.Module], None]],
                 train: Callable[[nn.Module, data.DataLoader, optim.Optimizer], float],
                 validation: Callable[[nn.Module, data.DataLoader], Tuple[float, float]],
                 inference: Callable[[List[Any]], List[float]],
                 train_loader: Callable[[], data.DataLoader],
                 val_loader: Callable[[], data.DataLoader],
                 optimizer: Callable[[nn.Module], optim.Optimizer]):

        self._network = network
        self._train = train
        self._init = init
        self._validation = validation
        self._inference = inference
        self._args = KubeArgs.parse()
        self._train_loader_func = train_loader
        self._val_loader_func = val_loader
        self._optimizer_func = optimizer

    def start(self):
        # if the args are train or validation, we should load the dataset
        pass

    def init(self):
        """
        Init applies the init function to the model to initialize
        in a specific way
        """
        self._network.apply(self._init)

    def train(self):
        """
        Train applies the train hook to all the data in the corresponding
        dataset
        :return:
        """

        # get the train loader thanks to the registered hooks
        # train for one epoch and get th result, by defaulr return the loss
        loader = self._train_loader_func()
        optimizer = self._optimizer_func(self._network)
        loss = self._train(self._network, loader, optimizer)
        pass

    def validate(self):
        loader = self._val_loader_func()
        acc, loss = self._validation(self._network, loader)
        pass

    def infer(self):
        pass
