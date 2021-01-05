import os

import numpy as np
import pymongo
from flask import Flask, request, jsonify
from .utils import *

app = Flask(__name__)

app.config['MONGO_ADDRESS'] = '192.168.99.101'
app.config['MONGO_PORT'] = 30933
app.config['UPLOAD_FOLDER'] = 'uploads'

client = pymongo.MongoClient(app.config['MONGO_ADDRESS'], app.config['MONGO_PORT'])


# Define the endpoints of the storage service
@app.route('/dataset/<string:name>', methods=['POST', 'DELETE'])
def handle_dataset(name: str):
    if request.method == 'POST':
        app.logger.debug('Received request to upload a dataset')
        return upload_dataset(name)

    elif request.method == 'DELETE':
        app.logger.debug("Received request to delete the dataset")
        return delete_dataset(name)


# Handles the upload of a dataset
# Sees if the file has an npy or pkl extension
# and according to that it divides the dataset in batches
# of constant size and saves them to the mongo database
def upload_dataset(name: str):
    if not request.files:
        app.logger.error('Request does not include a file')
        return jsonify(error='Request does not include a file'), 400

    app.logger.debug(f'handling dataset creation for dataset {name}')

    # save the file in the server and then split it and save
    # it in the database
    db_names = set(client.list_database_names())
    app.logger.debug(f'Db names {db_names}')
    if name in db_names:
        return jsonify(error=f'Dataset {name} already exists'), 400

    # get the file
    train_data = request.files['train']
    test_data = request.files['test']

    # # build the file names
    extension = train_data.filename.split('.')[-1]
    app.logger.debug(f'Extension is {extension}')
    tr_path = os.path.join(app.config['UPLOAD_FOLDER'], f'train.{extension}')
    te_path = os.path.join(app.config['UPLOAD_FOLDER'], f'test.{extension}')

    train_data.save(tr_path)
    test_data.save(te_path)

    # Process the datasets
    return _process_datasets(tr_path, te_path, extension)


def _process_datasets(train_path: str, test_path: str, extension: str):
    if extension not in ['npy', 'pkl']:
        return jsonify(error='File extension not suported, must be one of [npy, pkl]'), 400

    train_data, test_data = None, None

    if extension == 'npy':
        app.logger.debug('Loading npy files')
        train_data = np.load(train_path)
        test_data = np.load(test_path)

    elif extension == 'pkl':
        app.logger.debig('Loading pickle files')
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)

    return jsonify(train=list(train_data.shape), test=list(test_data.shape)), 200


def delete_dataset():
    pass


if __name__ == '__main__':
    app.run(debug=True)
