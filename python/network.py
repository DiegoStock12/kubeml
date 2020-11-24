""" Example file that will be run in the functions """

import time
# misc
from typing import Dict

# redis
import redisai as rai
# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata
# read the request params and logging
from flask import current_app, request

# some constants for testing
redis_addr = '192.168.99.102'
redis_port = 6379
ps_url = 'http://scheduler.default'

# parameters that we will have to set
# TODO maybe this should be inside a dataclass?
psId = None
funcId = None
psPort = None
task = None
N = None

# Set some global stuff
tensor_dict: Dict[str, torch.Tensor] = dict()  # Tensor to accumulate the gradients
redis_con = rai.Client(host=redis_addr, port=redis_port)


# Define the network that we'll use to train
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


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


def parse_url_args():
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
    global psId, psPort, N, task, funcId

    # Set the global variables
    psId = request.args.get('psId')
    psPort = request.args.get('psPort')
    N = request.args.get('N')
    task = request.args.get('task')
    funcId = request.args.get('funcId')

    current_app.logger.info(f'Loaded the configs: funcId={funcId}, N={N}, task={task}, psId={psId}, psPort={psPort}')


def load_model_weights(model: nn.Module):
    """Load the model weights saved in the database to start the new epoch"""
    current_app.logger.info('Loading model from database')
    with torch.no_grad():
        for name, layer in model.named_children():
            # only load and save layers that have bias
            # this excludes dropout and pool layers
            # since those weights are not optimizable
            if hasattr(layer, 'bias'):
                # How the layers are saved in the database
                weight_key = f'{psId}:{name}-weight'
                bias_key = f'{psId}:{name}-bias'

                current_app.logger.info(f'Loading weights and biases for layer {name}')

                # Load the particular layer weight and bias
                w = redis_con.tensorget(weight_key)
                layer.weight = torch.nn.Parameter(torch.from_numpy(w))

                w = redis_con.tensorget(bias_key)
                layer.bias = torch.nn.Parameter(torch.from_numpy(w))

    current_app.logger.info('Model loaded from database')


def save_model_weights(model: nn.Module):
    """After the init task we should save the model gradients to the database"""
    current_app.logger.info('Saving model to the database')
    with torch.no_grad():
        for name, layer in model.named_children():
            if hasattr(layer, 'bias'):
                weight_key = f'{psId}:{name}-weight'
                bias_key = f'{psId}:{name}-bias'

                current_app.logger.info(f'Setting weights and biases for layer {name}')

                # Save
                redis_con.tensorset(weight_key, layer.weight.cpu().detach().numpy(), dtype='float32')
                redis_con.tensorset(weight_key, layer.weight.cpu().detach().numpy(), dtype='float32')

    current_app.logger.info('Saved model to the database')


def save_gradients():
    """Save the gradients in the REDIS database after training"""

    for grad_name, tensor in tensor_dict.items():
        current_app.logger.info(f'Setting the gradients for {psId}:{grad_name}/{funcId}')
        redis_con.tensorset(f'{psId}:{grad_name}/{funcId}', tensor.cpu().numpy())

    current_app.logger.info('All the gradients were set in the db')


def create_model():
    """Creates the model used to train the network

    For this example we'll be using the simple model from the MNIST examples
    (https://github.com/pytorch/examples/blob/master/mnist/main.py)
    """

    def init_weights(m: nn.Module):
        """Initialize the weights of the network"""
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.01)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.01)

    # Create the model and initialize the weights
    model = Net()

    # If the task is initializing the layers do so
    if task == 'init':
        current_app.logger.info('Initializing layers...')
        model.apply(init_weights)

    return model


def train(model: nn.Module, device,
          train_loader: tdata.DataLoader,
          optimizer: torch.optim.Optimizer):
    """Loop used to train the network"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        # Here save the gradients to publish on the database
        update_tensor_dict(model, tensor_dict)
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                1, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))


def validate(model, device, val_loader: tdata.DataLoader):
    """Loop used to validate the network"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


# The function that will be run by default when the fission func is invoked
# TODO fill the rest of the function once we know how to load the data in Kubernetes
def main():
    start = time.time()
    current_app.logger.info(f'Started serving request')

    # 1) Parse args to see the kind of task we have to do
    parse_url_args()

    # build the model
    model: nn.Module = create_model()

    # If we just need to init the model save and exit
    if task == 'init':
        # Save the models and return the weights
        save_model_weights(model)
        return f'Model saved, layers are {[name for name, layer in model.named_children() if hasattr(layer, "bias")]}'

    if task == 'val':
        return f"""Task is validation, received parameters are 
                funcId={funcId}, N={N}, task={task}, psId={psId}, psPort={psPort}
                completed in {time.time() - start}"""

    return f"""Task is training, received parameters are 
                funcId={funcId}, N={N}, task={task}, psId={psId}, psPort={psPort}
                completed in {time.time() - start}"""
