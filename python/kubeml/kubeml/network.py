from abc import ABC, abstractmethod

import logging
import os
from typing import Dict, List, Tuple, Any, Union

import flask
import numpy as np
import redisai as rai
import torch
from flask import request, jsonify
from redis.exceptions import RedisError

from .dataset import _KubeArgs
from .exceptions import *
from .util import *

# Load from environment the values from th MONGO IP and PORT
try:
    REDIS_URL = os.environ['REDIS_URL']
    REDIS_PORT = os.environ['REDIS_PORT']
    logging.debug(f'Found configuration for redis {REDIS_URL}:{REDIS_PORT}')
except KeyError:
    logging.debug("Could not find redis configuration in env, using defaults")
    REDIS_URL = "redisai.kubeml"
    REDIS_PORT = 6379


class KubeModel(ABC):

    def __init__(self, network: nn.Module):
        self._network = network

        # initialize redis connection
        self._redis_client = rai.Client(host=REDIS_URL, port=REDIS_PORT)

    def start(self) -> Tuple[flask.Response, int]:
        """
        Start executes the function invoked by the user
        """
        self.args = _KubeArgs.parse()
        task = self.args._task

        if task == "init":
            layers = self.__initialize()
            return jsonify(layers), 200

        elif task == "train":
            loss = self.__train()
            return jsonify(loss=loss), 200

        elif task == "val":
            acc, loss = self.__validate()
            return jsonify(loss=loss, accuracy=acc), 200

        elif task == "infer":
            preds = self.__infer()
            return jsonify(predictions=preds), 200

        else:
            self._redis_client.close()
            raise KubeMLException(f"Task {task} not recognized", 400)

    def __initialize(self) -> List[str]:
        """
        Initializes the network

        :return: the names of the optimizable layers of the network, which will be saved in the reference model
        """
        try:
            self.init(self._network)
            self.__save_model()

        except RedisError as re:
            raise StorageError(re)
        finally:
            self._redis_client.close()

        return [name for name, layer in self._network.named_modules() if is_optimizable(layer)]

    # TODO if we want to implement K-AVG... we could tune it here or in the PS directly
    # TODO I think it is better that the PS chooses how big N is depending on the size of the dataset
    def __train(self) -> float:
        """
        Function called to train the network. Loads the reference model from the database,
        trains with the method provided by the user and saves the model after training to the database

        :return: The loss of the epoch, as returned by the user function
        """

        # Loads the model, train and save the results after returning the loss

        try:
            self.__load_model()
            loss = self.train(self._network)
            self.__save_model()
        except RedisError as re:
            raise StorageError(re)
        finally:
            self._redis_client.close()

        return loss

    def __validate(self):

        try:
            self.__load_model()
            acc, loss = self.validate(self._network)
        except RedisError as re:
            raise StorageError(re)
        finally:
            self._redis_client.close()

        return acc, loss

    def __infer(self) -> Union[torch.Tensor, np.ndarray, List[float]]:
        data_json = request.json
        if not data_json:
            logging.error("JSON not found in request")
            raise DataError

        preds = self.infer(self._network, data_json)

        if isinstance(preds, torch.Tensor):
            return preds.cpu().numpy().tolist()
        elif isinstance(preds, np.ndarray):
            return preds.tolist()
        elif isinstance(preds, list):
            return preds
        else:
            raise InvalidFormatError

    def __load_model(self):
        """
        Loads the model from redis ai and applies it to the network
        """
        state_dict = self.__get_model_dict()
        self._network.load_state_dict(state_dict)
        logging.debug("Loaded state dict from redis")

    def __save_model(self):
        """
        Saves the model to the tensor storage
        """
        job_id = self.args._job_id
        task = self.args._task
        func_id = self.args._func_id

        logging.debug("Saving model to the database")
        with torch.no_grad():
            for name, layer in self._network.named_modules():
                if is_optimizable(layer):
                    # Save the weights
                    logging.debug(f'Setting weights for layer {name}')
                    weight_key = f'{job_id}:{name}.weight' \
                        if task == 'init' \
                        else f'{job_id}:{name}.weight/{func_id}'
                    self._redis_client.tensorset(weight_key, layer.weight.cpu().detach().numpy(), dtype='float32')

                    # Save the bias if not None
                    if layer.bias is not None:
                        logging.debug(f'Setting bias for layer {name}')
                        bias_key = f'{job_id}:{name}.bias' \
                            if task == 'init' \
                            else f'{job_id}:{name}.bias/{func_id}'
                        self._redis_client.tensorset(bias_key, layer.bias.cpu().detach().numpy(), dtype='float32')

        logging.debug('Saved model to the database')

    def __get_model_dict(self) -> Dict[str, torch.Tensor]:
        """
        Fetches the model weights from the tensor storage

        :return: The state dict of the reference model
        """
        job_id = self.args._job_id

        state = dict()
        for name, layer in self._network.named_modules():
            if is_optimizable(layer):
                logging.debug(f"Loading weights for layer {name}")
                weight_key = f'{job_id}:{name}.weight'
                w = self._redis_client.tensorget(weight_key)
                # set the weight
                state[weight_key[len(job_id) + 1:]] = torch.from_numpy(w)

                # If the layer has an active bias retrieve it
                # Some of the layers in resnet do not have bias
                # or it is None. It is not needed with BN, so skip it
                if layer.bias is not None:
                    logging.debug(f'Loading bias for layer {name}')
                    bias_key = f'{job_id}:{name}.bias'
                    w = self._redis_client.tensorget(bias_key)
                    # set the bias
                    state[bias_key[len(job_id) + 1:]] = torch.from_numpy(w)

        logging.debug(f'Layers are {state.keys()}')

        return state

    @abstractmethod
    def init(self, model: nn.Module):
        pass

    @abstractmethod
    def train(self, model: nn.Module) -> float:
        pass

    @abstractmethod
    def validate(self, model: nn.Module) -> Tuple[float, float]:
        pass

    @abstractmethod
    def infer(self, model: nn.Module, data: List[Any]) -> Union[torch.Tensor, np.ndarray, List[float]]:
        pass
