"""Utils to work with saving and loading gradients"""

import redisai as rai
import torch
import torch.nn as nn
from dataclasses import dataclass
from flask import request, current_app

# some constants for testing
redis_addr = 'redisai.default'
redis_port = 6379
ps_url = 'http://scheduler.default'

redis_con = rai.Client(host=redis_addr, port=redis_port)


@dataclass
class TrainParams:
    ps_id: str
    ps_port: int
    N: int
    task: str
    func_id: int


def update_tensor_dict(m: nn.Module, d: dict):
    """Update the tensor dict so we can save it after the epoch is finished"""
    with torch.no_grad():
        for n, l in m.named_children():
            if hasattr(l, 'weight'):
                if n in d:
                    d[f'{n}-weight-grad'] += l.weight.grad
                    d[f'{n}-bias-grad'] += l.bias.grad
                else:
                    d[f'{n}-weight-grad'] = l.weight.grad
                    d[f'{n}-bias-grad'] = l.bias.grad


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
    TODO maybe we should add a link to the dataset uri so they can find it
    TODO or standardize that the dataset should be divided in test/train and easy
    """

    # Set the global variables
    ps_id = request.args.get('psId')
    ps_port = int(request.args.get('psPort'))
    N = int(request.args.get('N'))
    task = request.args.get('task')
    func_id = int(request.args.get('funcId'))

    current_app.logger.info(f'Loaded the configs: funcId={func_id}, N={N}, task={task}, psId={ps_id}, psPort={ps_port}')

    return TrainParams(
        ps_id=ps_id,
        ps_port=ps_port,
        N=N,
        task=task,
        func_id=func_id
    )


def load_model_weights(model: nn.Module, params: TrainParams):
    """Load the model weights saved in the database to start the new epoch"""
    current_app.logger.info('Loading model from database')
    with torch.no_grad():
        for name, layer in model.named_children():
            # only load and save layers that have bias
            # this excludes dropout and pool layers
            # since those weights are not optimizable
            if hasattr(layer, 'bias'):
                # How the layers are saved in the database
                weight_key = f'{params.ps_id}:{name}-weight'
                bias_key = f'{params.ps_id}:{name}-bias'

                current_app.logger.info(f'Loading weights and biases for layer {name}')

                # Load the particular layer weight and bias
                w = redis_con.tensorget(weight_key)
                layer.weight = torch.nn.Parameter(torch.from_numpy(w))

                w = redis_con.tensorget(bias_key)
                layer.bias = torch.nn.Parameter(torch.from_numpy(w))

    current_app.logger.info('Model loaded from database')


def save_model_weights(model: nn.Module, params: TrainParams):
    """After the init task we should save the model gradients to the database"""
    current_app.logger.info('Saving model to the database')
    with torch.no_grad():
        for name, layer in model.named_children():
            if hasattr(layer, 'bias'):
                weight_key = f'{params.ps_id}:{name}-weight'
                bias_key = f'{params.ps_id}:{name}-bias'

                current_app.logger.info(f'Setting weights and biases for layer {name}')

                # Save
                redis_con.tensorset(weight_key, layer.weight.cpu().detach().numpy(), dtype='float32')
                redis_con.tensorset(bias_key, layer.bias.cpu().detach().numpy(), dtype='float32')

    current_app.logger.info('Saved model to the database')


def save_gradients(tensor_dict: dict, params: TrainParams):
    """Save the gradients in the REDIS database after training"""

    for grad_name, tensor in tensor_dict.items():
        current_app.logger.info(f'Setting the gradients for {params.ps_id}:{grad_name}/{params.ps_id}')
        redis_con.tensorset(f'{params.ps_id}:{grad_name}/{params.func_id}', tensor.cpu().numpy())

    current_app.logger.info('All the gradients were set in the db')
