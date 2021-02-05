"""Utils to work with saving and loading gradients"""

import pickle
from typing import Dict

import redisai as rai
import torch
import torch.nn as nn
from dataclasses import dataclass
from flask import request, current_app
from pymongo import MongoClient

# some constants for testing
redis_addr = 'redisai.default'
redis_port = 6379
ps_url = 'http://scheduler.default'
ps_local = 'http://127.0.0.1'

# Constants to access the mongo database
KUBEML_DATABASE = "kubeml"
NETWORK_COLLECTION = 'network'
MONGO_IP = 'mongodb.default'
MONGO_PORT = 27017

redis_con = rai.Client(host=redis_addr, port=redis_port)


# These are the dataclasses that the PS sends to the function
# and the results sent back to the PS api
@dataclass
class TrainParams:
    ps_id: str
    N: int
    task: str
    func_id: int
    lr: float
    batch_size: int


# This is sent back to the parameter server
# TODO make a PS service instead of using the scheduler for everything
# results is a dictionary with the results from the training ( TODO Should be able to be defined
# by the user in the future ) like: results: { loss: final_loss, train_accuracy: train_acc }
# We'll just use the loss for now
# @dataclass
# class TrainResults:
#     results: dict


def update_tensor_dict(m: nn.Module, d: dict):
    """Update the tensor dict so we can save it after the epoch is finished"""
    with torch.no_grad():
        for name, layer in m.named_modules():
            if is_optimizable(layer):
                if name in d:
                    d[f'{name}.weight.grad'] += layer.weight.grad
                    if layer.bias is not None:
                        d[f'{name}.bias.grad'] += layer.bias.grad
                else:
                    d[f'{name}.weight.grad'] = layer.weight.grad
                    if layer.bias is not None:
                        d[f'{name}.bias.grad'] = layer.bias.grad


def parse_url_args() -> TrainParams:
    """The parameter server will send some arguments like the function id, the response port
    and the total number of functions as a query string. Parse all of these

    - funcId: number of the function that will determine the response and how we save the gradients
    - N: number of functions in the epoch. Will determine the amount of data to read from the storage
    - psPort: port where the parameter server will be waiting for the results
    - psId: Id of the parameter server that manages this function
    - task: type of task that the function should perform (train | val | init)
        - train is the normal function. Loads the dataset and optimizes the network
        - val just takes the validation dataset and returns its accuracy to the ps
        - init just initializes the network so that all the following workers use the same weights
    - lr: learning rate to be applied in this epoch
    - batch_size: internal batch size to be used # TODO check this
    TODO maybe we should add a link to the dataset uri so they can find it
    TODO or standardize that the dataset should be divided in test/train and easy
    """

    # Set the global variables
    ps_id = request.args.get('psId')
    N = int(request.args.get('N'))
    task = request.args.get('task')
    func_id = int(request.args.get('funcId'))
    batch = int(request.args.get('batchSize'))
    lr = float(request.args.get('lr'))

    current_app.logger.info(f'''Loaded the configs: funcId={func_id},
                            N={N}, task={task}, psId={ps_id},
                            batch-size={batch}, lr={lr}''')

    return TrainParams(
        ps_id=ps_id,
        N=N,
        task=task,
        func_id=func_id,
        batch_size=batch,
        lr=lr
    )


# def load_model_weights(model: nn.Module, params: TrainParams):
#     """Load the model weights saved in the database to start the new epoch"""
#     current_app.logger.info('Loading model from database')
#     with torch.no_grad():
#         for name, lafyer in model.named_modules():
#             # only load and save layers that are optimizable (conv or fc)
#             if _is_optimizable(layer):
#
#                 # Load the weight
#                 current_app.logger.info(f'Loading weights for layer {name}')
#                 weight_key = f'{params.ps_id}:{name}.weight'
#                 w = redis_con.tensorget(weight_key)
#                 layer.weight = torch.nn.Parameter(torch.from_numpy(w))
#
#                 # If the layer has an active bias retrieve it
#                 # Some of the layers in resnet do not have bias
#                 # or it is None. It is not needed with BN, so skip it
#                 if layer.bias is not None:
#                     current_app.logger.info(f'Loading bias for layer {name}')
#                     bias_key = f'{params.ps_id}:{name}.bias'
#                     w = redis_con.tensorget(bias_key)
#                     layer.bias = torch.nn.Parameter(torch.from_numpy(w))
#
#     current_app.logger.info('Model loaded from database')

def load_final_model(model: nn.Module, params: TrainParams):
    """Based on the parameters (taskId or id of the trained model),
    load the weights
    """
    # get the mongo client
    client = MongoClient(MONGO_IP, MONGO_PORT)
    db = client[KUBEML_DATABASE]
    result = db[NETWORK_COLLECTION].find_one({"_id": params.ps_id})
    if result is None:
        raise FileNotFoundError

    # load the statedict and pass to the model
    # make the loading in place so we do not return anything
    state = pickle.loads(result['state_dict'])
    model.load_state_dict(state)


def load_model_weights(model: nn.Module, params: TrainParams):
    """Load the model weights saved in the database to start the new epoch"""
    # Get the tensors as a state dict from the database
    state_dict = _get_model_dict(model, params.ps_id)

    # Load the tensors onto the model
    # By making it not strict we can omit layers in the state
    # dict if those are not optimizable
    model.load_state_dict(state_dict)

    current_app.logger.info(f'Loaded state dict from the database {state_dict.keys()}')


def _get_model_dict(model: nn.Module, ps_id: str) -> Dict[str, torch.Tensor]:
    """Loads all the tensors from the weights and the biases from the database.
    Parse them and introduce them in a dictionary that is then used by the torch
    load_state_dict method to load the weights and biases of the previous epochs"""
    state = dict()
    for name, layer in model.named_modules():
        # only load and save layers that are optimizable (conv or fc)
        if is_optimizable(layer):

            # Load the weight
            current_app.logger.debug(f'Loading weights for layer {name}')
            weight_key = f'{ps_id}:{name}.weight'
            w = redis_con.tensorget(weight_key)
            # set the weight
            state[weight_key[len(ps_id) + 1:]] = torch.from_numpy(w)

            # If the layer has an active bias retrieve it
            # Some of the layers in resnet do not have bias
            # or it is None. It is not needed with BN, so skip it
            if layer.bias is not None:
                current_app.logger.debug(f'Loading bias for layer {name}')
                bias_key = f'{ps_id}:{name}.bias'
                w = redis_con.tensorget(bias_key)
                # set the bias
                state[bias_key[len(ps_id) + 1:]] = torch.from_numpy(w)

    current_app.logger.debug(f'Layers are {state.keys()}')

    return state


# def save_model_weights(model: nn.Module, params: TrainParams):
#     """After the init task we should save the model gradients to the database"""
#     current_app.logger.info('Saving model to the database')
#     with torch.no_grad():
#         for name, layer in model.named_children():
#             if hasattr(layer, 'bias'):
#                 weight_key = f'{params.ps_id}:{name}-weight'
#                 bias_key = f'{params.ps_id}:{name}-bias'
#
#                 current_app.logger.info(f'Setting weights and biases for layer {name}')
#
#                 # Save
#                 redis_con.tensorset(weight_key, layer.weight.cpu().detach().numpy(), dtype='float32')
#                 redis_con.tensorset(bias_key, layer.bias.cpu().detach().numpy(), dtype='float32')
#
#     current_app.logger.info('Saved model to the database')

def save_model_weights(model: nn.Module, params: TrainParams):
    """Save the model weights to the database after each epoch
    If it is the init task we save the model as is without function
    identifier. In other cases we save the statedict of the function
    identified by its function ID"""
    current_app.logger.info('Saving model to the database')
    with torch.no_grad():
        for name, layer in model.named_modules():
            if is_optimizable(layer):

                # Save the weights
                current_app.logger.info(f'Setting weights for layer {name}')
                weight_key = f'{params.ps_id}:{name}.weight' \
                    if params.task == 'init' \
                    else f'{params.ps_id}:{name}.weight/{params.func_id}'
                redis_con.tensorset(weight_key, layer.weight.cpu().detach().numpy(), dtype='float32')

                # Save the bias if not None
                if layer.bias is not None:
                    current_app.logger.info(f'Setting bias for layer {name}')
                    bias_key = f'{params.ps_id}:{name}.bias' \
                        if params.task == 'init' \
                        else f'{params.ps_id}:{name}.bias/{params.func_id}'
                    redis_con.tensorset(bias_key, layer.bias.cpu().detach().numpy(), dtype='float32')

    current_app.logger.info('Saved model to the database')


def is_optimizable(layer: nn.Module) -> bool:
    """Should save layer returns just whether the layer is optimizable or not
    and thus if it should be sent to the parameter server"""
    t = str(type(layer))
    if 'conv' in t or 'linear' in t:
        return True
    return False


def save_gradients(tensor_dict: dict, params: TrainParams):
    """Save the gradients in the REDIS database after training"""

    for grad_name, tensor in tensor_dict.items():
        current_app.logger.info(f'Setting the gradients for {params.ps_id}:{grad_name}/{params.func_id}')
        redis_con.tensorset(f'{params.ps_id}:{grad_name}/{params.func_id}', tensor.cpu().numpy())

    current_app.logger.info('All the gradients were set in the db')


# Send the train results back to the PS
# TODO ps should be its own service
def send_train_finish(params: TrainParams, **kwargs) -> dict:
    """With the train params build the url and build the response"""
    # for now just take the kwargs and that's it

    # Simply return whatever the user chooses
    # by calling this function with loss: smthg, acc: smthg, those are
    # returned
    res_d = kwargs
    current_app.logger.info(f"Sending response {res_d}")

    # build the url and post the results
    # url = f'{ps_local}:{params.ps_port}/finish/{params.func_id}'
    # TODO uncomment this
    # r = requests.post(url, json=res_d)

    # if r.status_code != 200:
    #     current_app.logger.error(f'Error sending results to the server {r.status_code}')

    return res_d


def clean(data, model):
    """Simply deletes the current con and dataset and model"""
    del data, model
    redis_con.close()