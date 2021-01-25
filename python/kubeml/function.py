from typing import Callable, Dict, List

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class Network:

    def __init__(self,
                 model: nn.Module,
                 init: Callable[[nn.Module], None],
                 train: Callable[[data.DataLoader, optim.Optimizer], Dict[str, float]],
                 validation: Callable[[data.DataLoader], Dict[str, float]],
                 inference: Callable[[List[float]], List[float]]):

        self._train = train
        self._init = init
        self._validation = validation
        self._inference = inference
        self._model = model

        self._K = 64

    def start(self):
        pass

    def init(self):
        self._model.apply(self._init)

    def train(self) -> Dict[str, float]:
        for _ in range(self._K):
            pass

        return {}

    def validate(self) -> Dict[str, float]:
        return {}

    def infer(self) -> List[float]:
        return []
