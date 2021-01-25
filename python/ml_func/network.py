""" Example file that will be run in the functions """

import time
# misc
from typing import Dict

import ml_dataset
import numpy as np
# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tdata
import torchvision.transforms as transforms
# Flask and logging
from flask import current_app, jsonify, request

# import the utils and dataset
import train_utils

# params of the training
train_params: train_utils.TrainParams = None


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
    if train_params.task == 'init':
        current_app.logger.info('Initializing layers...')
        model.apply(init_weights)

    return model


def train(model: nn.Module, device,
          train_loader: tdata.DataLoader,
          optimizer: torch.optim.Optimizer,
          tensor_dict: Dict[str, torch.Tensor]) -> float:
    """Loop used to train the network"""
    model.train()
    loss = None
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()

        # Here save the gradients to publish on the database
        # train_utils.update_tensor_dict(model, tensor_dict)
        optimizer.step()

        if batch_idx % 4 == 0:
            current_app.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                1, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))

    return loss.item()


def validate(model,
             device,
             val_loader: tdata.DataLoader) -> (float, float):
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

    accuracy = 100. * correct / len(val_loader.dataset)
    current_app.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return accuracy, test_loss


def infer(model, device, data: np.array, transform):
    """ Return the predictions for the sent datapoints"""
    model.eval()
    data = transform(data).to(device)
    data = data.permute(1, 2, 0).view(-1, 1, 28, 28)
    out = model(data)

    preds = torch.argmax(out, axis=1)
    return preds.cpu().numpy()


# The function that will be run by default when the fission func is invoked
# TODO fill the rest of the function once we know how to load the data in Kubernetes
def main():
    global train_params

    # Create the tensor dict for this model
    tensor_dict: Dict[str, torch.Tensor] = dict()

    start = time.time()
    current_app.logger.info(f'Started serving request')

    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_app.logger.info(f'Running on device {device}')

    # 1) Parse args to see the kind of task we have to do
    train_params = train_utils.parse_url_args()

    # Create the transformation
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # build the model
    model: nn.Module = create_model()

    # If we just want to init save the model and return
    # TODO should only parse all train_params if we don't have inference
    if train_params.task == 'init':
        # Save the models and return the weights
        train_utils.save_model_weights(model, train_params)
        return jsonify([name for name, layer in model.named_children() if hasattr(layer, "bias")])

    # Perform an inference task
    # In these cases we will extract the data from the request body and
    # pass it to the model
    # TODO check how feasible it is to do everything from JSON
    elif train_params.task == 'infer':
        # get the requests
        # the predictions will have the shape of
        # a json object with an array of datapoints, these
        # datapoints can be of any shape
        data_json = request.json
        if not data_json:
            current_app.logger.error('JSON not found in request')
            return jsonify(error="Not found"), 400

        # Load the model with the latest weights
        train_utils.load_final_model(model, train_params)

        # parse the numpy array from the request
        data = np.array(data_json["data"], dtype='uint8')
        current_app.logger.debug(f'Shape of the data is {data.shape}')
        preds = infer(model, device, data, transf)
        return jsonify(predictions=preds.tolist()), 200

    # For training or validation we need to
    # 1) create the dataset
    # 2) load the model weights
    # 3) train or validate
    # (if we train) publish the gradients on the cache
    dataset = ml_dataset.MnistDataset(func_id=train_params.func_id, num_func=train_params.N,
                                      task=train_params.task, transform=transf)
    train_utils.load_model_weights(model, train_params)
    # TODO receive the batch size through the api call
    loader = tdata.DataLoader(dataset, batch_size=train_params.batch_size)
    current_app.logger.info(f'built dataset of size {dataset.data.shape} task is {train_params.task}')

    # If we want to validate we call test, if not we call train, we return the stats from the
    if train_params.task == 'val':
        acc, loss = validate(model, device, loader)
        res = train_utils.send_train_finish(train_params, loss=loss, accuracy=acc)
        current_app.logger.info(f"""Task is validation, received parameters are 
                funcId={train_params.func_id}, N={train_params.N}, task={train_params.task}, 
                psId={train_params.ps_id},
                completed in {time.time() - start}""")
        current_app.logger.info(f'Loaded model bias {model.fc2.bias}')
        return jsonify(res)


    # Train the network
    # Create an optimizer and do a full epoch on the parts of the data that
    # matter
    elif train_params.task == 'train':
        optimizer = optim.Adam(model.parameters(), lr=train_params.lr)
        loss = train(model, device, loader, optimizer, tensor_dict)

        # After training save the weights in the database
        train_utils.save_model_weights(model, train_params)
        res = train_utils.send_train_finish(train_params, loss=loss)
        current_app.logger.info(f"""Task is training, received parameters are 
                funcId={train_params.func_id}, N={train_params.N}, task={train_params.task}, 
                psId={train_params.ps_id},
                completed in {time.time() - start}, res={res}""")
        return jsonify(res)
