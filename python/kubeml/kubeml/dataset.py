import logging
import os
import pickle
from abc import ABC

import numpy as np
import torch.utils.data as data
from flask import request
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from .exceptions import *
from .util import *

# Load from environment the values from th MONGO IP and PORT
try:
    MONGO_URL = os.environ['MONGO_IP']
    MONGO_PORT = os.environ['MONGO_PORT']
    logging.debug(f'Found configuration for storage {MONGO_URL}:{MONGO_PORT}')
except KeyError:
    logging.debug("Could not find mongo configuration in env, using defaults")
    MONGO_URL = "mongodb.kubeml"
    MONGO_PORT = 27017


class _KubeArgs:
    """
    Arguments used by the function to transmit the information needed to perform a
    training, validation or inference task. The arguments are:
    """

    def __init__(self, job_id: str,
                 N: int, K: int,
                 task: str,
                 func_id: int,
                 lr: float = 0, batch_size: int = 0):
        """
        :arg job_id: id of the job\n
        :arg N: number of functions or parallelism
        :arg K: parameter for K-averaging, number of forward passes before sync
        :arg task: type of task (train, inference...)
        :arg func_id: id of the function
        :arg lr: learning rate
        :arg batch_size: size of the batch
        """

        self._job_id = job_id
        self._N = N
        self._K = K
        self._task = task
        self._func_id = func_id
        self.lr = lr
        self.batch_size = batch_size

    @classmethod
    def parse(cls):
        """
        Parses the arguments from the request context
        :return: returns a KubeArgs object used by other methods
        """
        try:
            job_id = request.args.get("jobId")
            N = request.args.get("N", type=int)
            K = request.args.get("K", type=int)
            task = request.args.get("task")
            func_id = request.args.get("funcId", type=int)
            lr = request.args.get("lr", type=float)
            batch_size = request.args.get("batchSize", type=int)

        except ValueError as ve:
            logging.error(f"Error parsing request arguments: {ve}, args:{request.args}")
            raise InvalidArgsError(ve)

        args = cls(job_id, N, K, task, func_id, lr, batch_size)
        return args


class KubeDataset(data.Dataset, ABC):
    """
    KubeDataset is the main abstraction used by KubeML to load the data in a
    distributed manner from the storage in Kubernetes.
    The datasets created by the users need to override the class and call the super() in the init.
    After that, they must override the __getitem__ method like in a normal torch dataset
    The Kube Dataset will expose two properties, data and labels, which will be automatically
    loaded from the database
    """

    def __init__(self, dataset: str):
        """
        Init reads the data from the database given the dataset name

        :arg dataset Name of the dataset in the KubeML storage service
        """

        self.dataset = dataset
        self._client = MongoClient(MONGO_URL, MONGO_PORT)
        self._database = self._client[dataset]
        self._args = None

        # data and labels of the dataset
        self.data, self.labels = None, None

        # Check first if the dataset that the user gave as input
        # is available in the configured storage service
        try:
            dbs = set(self._client.list_database_names())
            if self.dataset not in dbs:
                logging.error(f"Dataset not in the storage service. Dataset = {dataset},"
                              f"Available = {dbs}")
                self._client.close()
                raise DatasetNotFoundError

        except PyMongoError as e:
            self._client.close()
            raise StorageError(e)

        # Set the range of minibatches that this function will train on
        self.num_docs = self._database["train"].count_documents({})

    def _load_train_data(self, start: int, end: int):
        """
        For K averaging the data needs to be refreshed with the next K batches
        after every synchronization step, this is triggered by the KubeModel before
        starting another iteration

        :param start: first subset to be loaded
        :param end: last subset to be loaded
        """
        # load the minibatches given by the network
        minibatches = range(start, end)
        logging.debug(f"Loading minibatches {minibatches}")
        self.data, self.labels = self.__load_data(minibatches)

    def _load_validation_data(self):
        self.data, self.labels = self.__load_data()

    def _close(self):
        self._client.close()

    def __load_data(self, minibatches: range = None):

        # If the minibatches are None that means we have
        # to perform the validation so we load all documents in the
        # test collection
        #
        # If not, load the minibatches that belong to the function
        # given the number of functions N and the function id

        try:
            if minibatches is None:
                batches = self._database["test"].find({})
            else:
                batches = self._database["train"].find({
                    '_id': {'$gte': minibatches.start, '$lte': minibatches.stop - 1}
                })
        except PyMongoError as e:
            self._client.close()
            raise StorageError(e)

        data, labels = None, None
        for batch in batches:
            d = pickle.loads(batch['data'])
            l = pickle.loads(batch['labels'])

            if data is None:
                data, labels = d, l
            else:
                data = np.vstack([data, d])
                labels = np.hstack([labels, l])

        return data, labels.flatten()
