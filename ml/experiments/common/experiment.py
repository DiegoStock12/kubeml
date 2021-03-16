from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses_json import dataclass_json, Undefined, CatchAll
from typing import List, Dict, Any
import subprocess
import time

import pandas as pd

from .utils import check_stderr

kubeml = '/mnt/c/Users/diego/CS/thesis/ml/pkg/kubeml-cli/kubeml'


@dataclass_json
@dataclass
class TrainOptions:
    default_parallelism: int
    static_parallelism: bool
    validate_every: int
    k: int
    goal_accuracy: float


@dataclass_json
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
    options: TrainOptions


@dataclass_json
@dataclass
class TrainMetrics:
    validation_loss: List[float]
    accuracy: List[float]
    train_loss: List[float]
    parallelism: List[int]
    epoch_duration: List[float]


@dataclass_json
@dataclass
class History:
    id: str
    task: TrainRequest
    data: TrainMetrics


class Experiment(ABC):

    def __init__(self, title: str):
        self.title = title

    @abstractmethod
    def run(self):
        pass


class KubemlExperiment(Experiment):
    def __init__(self, title, request: TrainRequest):
        super(KubemlExperiment, self).__init__(title=title)
        self.request = request

        # Network ID is created when task is started through the CLI
        self.network_id = None
        self.history = None

    def run(self):
        """ RUn an experiment on KubeML

        - create the train task
        - watch until it finishes
        - load the history
        - TODO save the history somewhere
        """
        self.network_id = self.run_task()
        time.sleep(30)
        self.wait_for_task_finished()
        self.history = self.get_model_history()

        print(self.history.to_json())

        # TODO save the history in the file related to the experiment title

    def wait_for_task_finished(self):
        while True:
            done = self.check_if_task_finished()
            if done:
                return
            time.sleep(10)

    def run_task(self) -> str:
        """ Runs a task and returns the id assigned by kubeml"""
        command = f"{kubeml} train  \
                    --function {self.request.function_name} \
                    --dataset {self.request.dataset} \
                    --epochs {self.request.epochs} \
                    --batch {self.request.batch_size} \
                    --lr {self.request.lr} \
                    --default-parallelism {self.request.options.default_parallelism} \
                    --goal-accuracy {self.request.options.goal_accuracy} \
                    --K {self.request.options.k} \
                    --validate-every {self.request.options.validate_every} \
                    --static"

        print("starting training with command", command)

        res = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        check_stderr(res)

        id = res.stdout.decode().strip()

        print("Received id", id)
        return id

    def check_if_task_finished(self) -> bool:
        """Check if the task is the the list of running tasks"""
        command = f"{kubeml} task list --short"
        print("Checking running tasks", command)

        res = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        check_stderr(res)

        # get all the tasks running
        tasks = res.stdout.decode().splitlines()

        for id in tasks:
            print(id, end=' ')
            if id == self.network_id:
                print()
                return False
        return True

    def get_model_history(self) -> History:
        """Gets the training history for a certain model"""
        command = f"{kubeml} history get --network {self.network_id}"
        print("Getting model history with command", command)

        res = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        check_stderr(res)

        print("got history", res.stdout.decode())

        # decode the json to the history
        h = History.from_json(res.stdout.decode())

        print(h)
        return h

    def to_dataframe(self) -> pd.DataFrame:
        """Converts this experiment to a pandas dataframe"""
        if not self.history:
            return None

        # simply flatten the dict and convert it into a dataframe
        # for this we use the to_dict func of the dataclass_json object
        flattened = {
            "id": self.history.id,
            **self.history.task.to_dict(),
            **self.history.task.options.to_dict()
        }
        del flattened['options']

        # add the arrays wrapped in a list so they all are considered
        # as 1 value, so all arrays are the same length
        for k, v in self.history.data.to_dict().items():
            flattened[k] = [v]

        return pd.DataFrame(flattened)

    def save(self, path: str) -> None:
        """Saves the experiment with the id as name in pickle format and as a pandas daframe"""

        # convert it to dataframe
        d = self.to_dataframe()
        _path = f'{path.rstrip("/")}/{self.network_id}.pkl'
        d.to_pickle(_path)

    def __str__(self):
        return f'KubeMLExperiment(title:{self.title})'


class TensorflowExperiment(Experiment):
    pass
