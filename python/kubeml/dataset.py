import torch.utils.data as data
from pymongo import MongoClient
from flask import request
import numpy as np
import pickle
import os
import logging
from typing import List, Generator

# Load from environment the values from th MONGO IP and PORT
try:
    MONGO_IP = os.environ['MONGO_IP']
    MONGO_PORT = os.environ['MONGO_PORT']
except KeyError:
    logging.error("Could not find mongo configuration in env, using defaults")
    MONGO_IP = "192.168.99.101"
    MONGO_PORT = 30933


class KubeArgs:
    """
    Arguments used by the function to transmit the information needed to perform a
    training, validation or inference task
    """

    def __init__(self, job_id: str,
                 N: int, task: str,
                 func_id: int,
                 lr: float = 0, batch_size: int = 0):
        self.job_id = job_id
        self.N = N
        self.task = task
        self.func_id = func_id
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
            task = request.args.get("task")
            func_id = request.args.get("funcId", type=int)
            lr = request.args.get("lr", type=float)
            batch_size = request.args.get("batchSize", type=int)

        except (ValueError, Exception) as ve:
            logging.error(f"Error parsing request arguments: {ve}")
            raise ve

        args = cls(job_id, N, task, func_id, lr, batch_size)
        return args


class KubeDataset(data.Dataset):
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

        self._client = MongoClient(MONGO_IP, MONGO_PORT)
        self._database = self._client[dataset]
        self._args = KubeArgs.parse()
        logging.debug(f"Parsed request args {self._args}")

        if self._args.task == "train":
            num_docs = self._database["train"].count_documents({})
            minibatches = self.__split_minibatches(range(num_docs), self._args.N)[num_docs]
            logging.debug("I get minibatches", minibatches)
            self.data, self.labels = self.__load_data(minibatches)

        else:
            self.data, self.labels = self.__load_data()

    def __load_data(self, minibatches: Generator[int, None, None] = None):

        # If the minibatches are None that means we have
        # to perform the validation so we load all documents in the
        # test collection
        #
        # If not, load the minibatches that belong to the function
        # given the number of functions N and the function id
        if minibatches is None:
            batches = self._database["test"].find({})
        else:
            batches = self._database["train"].find({
                '_id': {'$gte': minibatches.start, '$lte': minibatches.stop - 1}
            })

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

    @staticmethod
    def __split_minibatches(a, n) -> List[Generator[int, None, None]]:
        """
        Based on the number of minibatches return the ones assigned to each
        function so that the count is approximately the same
        """
        k, m = divmod(len(a), n)
        return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
