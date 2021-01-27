import pickle
import numpy as np
import pymongo
import torch.utils.data as tdata
from flask import current_app


MONGO_IP = 'mongodb.default'
MONGO_PORT = 27017
# TODO database should be passed as a parameter
DATABASE = 'mnist'


class MnistDataset(tdata.Dataset):
    """ The dataset is able to load a specific subset of samples from MONGO and combine them """

    def __init__(self, func_id, num_func, task, transform=None):
        """ Based on the funcId and the Number of functions, estimate the range of the datasets
        that we should get from mongo.

        :arg func_id ID of the function creating the dataset
        :arg num_func total number of functions invoked
        :arg task either train or val
        :arg transform transformations to be applied
        """

        # create the mongo client
        self.client = pymongo.MongoClient(MONGO_IP, MONGO_PORT)
        self.db = self.client[DATABASE]

        # get the number of documents
        ndocs = self.db.train.count_documents({})

        if task == 'train':
            minibatch = self._split_minibatches(range(ndocs), num_func)[func_id]
            print('I get minibatches', minibatch)

            self.data, self.labels = self._load_data(minibatch)

        # If task is validation we just tell it to load all the data
        else:
            self.data, self.labels = self._load_data()

        self.transforms = transform

    def __len__(self):
        return len(self.data)

    # We could have 1 document per datapoint or group of datapoints?
    # this is just useful for bigger datasets, but could save a lot of memory
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transforms:
            return self.transforms(x), y.astype('int64')
        else:
            return x, y.astype('int64')

    # TODO look how we can maybe declare indexes in mongodb to speed this up
    def _load_data(self, minibatches=None):
        """Return the data from the MongoDB already formatted as a numpy array"""

        if minibatches is None:
            # load all because it is the validation
            batches = self.db.test.find({})
        else:
            # Load the objects from Mongo
            batches = self.db.train.find({
                '_id': {'$gte': minibatches.start, '$lte': minibatches.stop-1}
            })

        data, labels = None, None
        for batch in batches:
            # Load the data and the labels and append it to the variables above
            d = pickle.loads(batch['data'])
            l = pickle.loads(batch['labels'])

            if data is None:
                data, labels = d, l
            else:
                current_app.logger.debug(f'Data is {data.shape}, labels are {labels.shape}')
                data = np.vstack([data, d])
                labels = np.hstack([labels, l])

        return data, labels.flatten()

    @staticmethod
    def _split_minibatches(a, n):
        """Based on the number of minibatches return the ones assigned to each
        function so that the count is approximately the same """
        k, m = divmod(len(a), n)
        return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]