from dataclasses import dataclass
import requests

from tensorflow.keras.callbacks import History as KerasHistory

from .experiment import Experiment, API_URL
from .utils import *


@dataclass
class TfConfig:
    network: str
    batch: int
    epochs: int


from typing import Callable


class TensorflowExperiment(Experiment):

    def __init__(self, config: TfConfig, main_func: Callable):
        super(TensorflowExperiment, self).__init__(self._get_title(config))
        self.title = self._get_title(config)
        self.history: KerasHistory = None
        self.func = main_func
        self.config = config
        self.hash = get_hash(self.title)

    def run(self):
        """based on the network and the parameters call one of the mains from the lenet or resnet funcs"""

        if self.config.network not in ('lenet', 'resnet'):
            print('Unknown network name', self.config.network)
            raise Exception('unknown network name')

        # start gathering the metrics
        self.start_metrics_collection()

        if self.config.network == 'lenet':
            self.history, self.times = self.func(self.config.epochs, self.config.batch)
            print('Lenet exp finished', self.config)

        elif self.config.network == 'resnet':
            self.history, self.times = self.func(self.config.epochs, self.config.batch)
            print('Resnet exp finished', self.config)

        # finish metrics
        self.end_metrics_collection()

    def save(self, path: str):
        d = self.to_dataframe()
        _path = f'{path.rstrip("/")}/{self.hash}.pkl'
        print('saving to', _path)
        d.to_pickle(_path)

    def to_dataframe(self) -> pd.DataFrame:

        d = {
            'model': self.config.network,
            'hash': self.hash,
            'batch_size': self.config.batch,
            'epochs': self.config.epochs,
            **{k: [v] for k, v in self.history.history.items()},
            'times': [self.times]
        }

        return pd.DataFrame(d)

    def start_metrics_collection(self):
        """Triggers the metrics api endpoint to start collecting metrics before the experiment"""
        url = API_URL + f'/new/{self.hash}'
        print('triggering start url...')

        resp = requests.put(url)
        if not resp.ok:
            print('error starting metrics')
        else:
            print('metrics collection started')

    @staticmethod
    def end_metrics_collection():
        """Triggers the api to stop collecting metrics"""
        print('triggering stop url...')
        url = API_URL + '/finish'
        resp = requests.delete(url)
        if not resp.ok:
            print('error stopping experiment')
        else:
            print("stopped experiment")

    @staticmethod
    def _get_title(config: TfConfig) -> str:
        return f'{config.network}-batch{config.batch}-epochs{config.epochs}'
