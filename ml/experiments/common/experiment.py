from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses_json import dataclass_json, Undefined, CatchAll
from typing import List, Dict, Any
import subprocess
import time
import os
import requests

import pandas as pd

from .utils import check_stderr, get_title, get_hash

kubeml = '../pkg/kubeml-cli/kubeml'


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


API_URL = 'http://localhost:5000'


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
        """
        self.network_id = self.run_task()
        time.sleep(5)

        # self.network_id = get_hash(self.title)

        # start collecting metrics from the experiments using the api
        self.start_metrics_collection()
        print('Training', end='', flush=True)
        self.wait_for_task_finished()

        print('Task finished, getting model history')
        self.history = self.get_model_history()

        # print(self.history.to_json())
        print('Experiment', self.network_id, 'finished')
        self.end_metrics_collection()

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

        done = False
        res = None
        for i in range(3):
            try:
                res = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                check_stderr(res)
                done = True
                break
            except Exception as _:
                print('error getting tasks, retrying')
                time.sleep(2)

        if not done:
            exit(-1)

        # get all the tasks running
        tasks = res.stdout.decode().splitlines()

        for id in tasks:
            print(id)
            if id == self.network_id:
                print('.', end='', flush=True)
                return False
        print()
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

    def start_metrics_collection(self):
        """Triggers the metrics api endpoint to start collecting metrics before the experiment"""
        url = API_URL + f'/new/{self.network_id}'
        print('triggering start url...')

        resp = requests.put(url)
        if not resp.ok:
            print('error starting metrics')
        else:
            print('metrics collection started')

    def end_metrics_collection(self):
        """Triggers the api to stop collecting metrics"""
        print('triggering stop url...')
        url = API_URL + '/finish'
        resp = requests.delete(url)
        if not resp.ok:
            print('error stopping experiment')
        else:
            print("stopped experiment")

    def _fake_history(self) -> History:

        self.history = History(
            id=self.network_id,
            task=self.request,
            data=TrainMetrics(
                validation_loss=[1],
                accuracy=[1],
                train_loss=[1],
                parallelism=[1],
                epoch_duration=[1]
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Converts this experiment to a pandas dataframe"""
        if not self.history:
            return None

        # simply flatten the dict and convert it into a dataframe
        # for this we use the to_dict func of the dataclass_json object
        flattened = {
            "id": self.history.id,
            "hash": get_hash(self.title),
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
        print('saving to', _path)
        d.to_pickle(_path)

    def __str__(self):
        return f'KubeMLExperiment(title:{self.title})'


class TensorflowExperiment(Experiment):
    pass
