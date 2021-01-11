import pickle
import logging
from pymongo import collection


def dataset_splits(data, labels, batch_size):
    """ Given the data, return constantly sized
    batches of the dataset, which will be saved to the
    database"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size], labels[i:i + batch_size]


def save_batches(col: collection.Collection, batches):
    """Saves the batches to the specified collection
    in the database"""
    logging.debug(f'Saving documents to the {col.full_name} collection')
    ids = col.insert_many([
        {'_id': i,
         'data': pickle.dumps(data, pickle.HIGHEST_PROTOCOL),
         'labels': pickle.dumps(labels, pickle.HIGHEST_PROTOCOL)
         }
        for i, (data, labels) in enumerate(batches)
    ]).inserted_ids
    logging.debug(f'Inserted {len(ids)} documents')
