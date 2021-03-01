import logging
import os
import pickle
import uuid

import numpy as np
import pymongo
from flask import Flask, request, jsonify
from utils import *

app = Flask(__name__)

# Check if the debug environment is present
try:
    debug = os.environ['DEBUG_ENV']
except KeyError:
    debug = False

# TODO add this also in an environment variable and put it in the yaml files
if debug:
    app.config['MONGO_ADDRESS'] = '192.168.99.101'
    app.config['MONGO_PORT'] = 30933
else:
    app.config['MONGO_ADDRESS'] = 'mongodb.kubeml'
    app.config['MONGO_PORT'] = 27017

app.config['UPLOAD_FOLDER'] = 'uploads'

# set some basic logging params
FORMAT = '[%(asctime)s] %(levelname)-8s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

# mongo connection
client = pymongo.MongoClient(app.config['MONGO_ADDRESS'], app.config['MONGO_PORT'])


@app.route('/health')
def health():
    return '', 200


# Define the endpoints of the storage service
@app.route('/dataset/<string:name>', methods=['POST', 'DELETE'])
def handle_dataset(name: str):
    if request.method == 'POST':
        logging.debug('Received request to upload a dataset')
        return upload_dataset(name)

    elif request.method == 'DELETE':
        logging.debug("Received request to delete the dataset")
        return delete_dataset(name)


# Handles the upload of a dataset
# Sees if the file has an npy or pkl extension
# and according to that it divides the dataset in batches
# of constant size and saves them to the mongo database
def upload_dataset(dataset_name: str):
    if not request.files:
        logging.error('Request does not include a file')
        return jsonify(error='Request does not include a file'), 400

    logging.debug(f'handling dataset creation for dataset {dataset_name}')

    # save the file in the server and then split it and save
    # it in the database
    db_names = set(client.list_database_names())
    logging.debug(f'Db names {db_names}')
    if dataset_name in db_names:
        logging.error(f"Dataset {dataset_name} already exists")
        return jsonify(error=f'Dataset {dataset_name} already exists'), 400

    file_names = list(request.files.keys())
    logging.debug(f'Files {file_names}')

    # create the unique identifier for the upload task
    # this prevents multiple concurrent uploads from overwriting each other
    upload_id = str(uuid.uuid4())[:8]
    logging.debug(f"Created upload id {upload_id}")

    # for each of the files (should be 4), load them
    # and save them to the database.
    # The files will be x-train, y-train, x-test, y-test
    for datatype in ['train', 'test']:
        logging.debug(f'Loading {datatype} data')

        # Load the features and the labels
        x = request.files[f'x-{datatype}']
        y = request.files[f'y-{datatype}']
        extension = x.filename.split('.')[-1]
        logging.debug(f'Extension is {extension}')

        # save the files to disk
        # save the data as x-train.ext, y-train.ext and so on
        x.save(os.path.join(app.config['UPLOAD_FOLDER'], f'x-{datatype}-{upload_id}.{extension}'))
        y.save(os.path.join(app.config['UPLOAD_FOLDER'], f'y-{datatype}-{upload_id}.{extension}'))
        logging.debug(f'Saved the {datatype} datasets to internal storage')

    # Process the datasets
    return _process_datasets(dataset_name, extension, upload_id)


def _process_datasets(dataset_name: str, extension: str, upload_id: str):
    if extension not in ['npy', 'pkl']:
        return jsonify(error='File extension not supported, must be one of [npy, pkl]'), 400

    data, targets = None, None

    for datatype in ['train', 'test']:

        x_path = os.path.join(app.config['UPLOAD_FOLDER'], f'x-{datatype}-{upload_id}.{extension}')
        y_path = os.path.join(app.config['UPLOAD_FOLDER'], f'y-{datatype}-{upload_id}.{extension}')

        if extension == 'npy':
            logging.debug('Loading npy files')
            data, targets = np.load(x_path), np.load(y_path)

        elif extension == 'pkl':
            logging.debug('Loading pickle files')
            with open(x_path, 'rb') as f:
                data = pickle.load(f)
            with open(y_path, 'rb') as f:
                targets = pickle.load(f)

        # for each of the targets and labels, join them and save them to the database
        # create the database for the dataset
        # generate the splits of constant size that will be used in the dataset
        # save the splits to the collection
        logging.debug(f'Saving the collection for {datatype} data')
        db = client[dataset_name]
        db.create_collection(datatype)

        splits = dataset_splits(data, targets, 64)
        save_batches(db[datatype], splits)

        # delete the documents from the server
        os.remove(x_path)
        os.remove(y_path)

    return jsonify(result='Dataset created'), 200


def delete_dataset(dataset_name: str):
    # Simply check that the dataset exists, and if so, delete it
    db_names = set(client.list_database_names())
    logging.debug(f'Datasets {db_names}')

    if dataset_name in db_names:
        logging.debug('Deleting dataset')
        client.drop_database(dataset_name)
        return jsonify(result='Dataset deleted'), 200

    logging.error("Dataset does not exist")
    return jsonify(error='Dataset does not exist'), 404


if __name__ == '__main__':
    app.run(debug=True)
