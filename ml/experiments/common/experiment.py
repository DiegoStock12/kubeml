from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class TrainRequest:
    """ Request holding the settings to run an experiment
    in Kubeml"""
    model_type: str
    batch_size: int
    epochs: int
    dataset: str
    lr: float
    function_name: str


class Experiment(ABC):

    def __init__(self, title: str):
        self.title = title

    @abstractmethod
    def run(self):
        pass


class KubemlExperiment(Experiment):
    def __init__(self, title, tasks: List[TrainRequest]):


class TensorflowExperiment(Experiment):
    pass
