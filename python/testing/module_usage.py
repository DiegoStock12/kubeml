from typing import List, Any, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.dataset import T_co
from flask import current_app

from kubeml.kubeml import KubeDataset, KubeModel

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


class KubeNet(KubeModel):

    def __init__(self, network: nn.Module):
        super().__init__(network)

        self.transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def train(self, model: nn.Module) -> float:
        current_app.logger.info("In the train function")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        current_app.logger.info(f'Using device {device}')
        dataset = ExampleDataset(transform=self.transf)
        train_loader = tdata.DataLoader(dataset, batch_size=self.args.batch_size)
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

        model = model.to(device)

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

    def validate(self, model: nn.Module) -> Tuple[float, float]:
        current_app.logger.info("In the validation function")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = ExampleDataset(transform=self.transf)
        val_loader = tdata.DataLoader(dataset, batch_size=self.args.batch_size)

        model = model.to(device)

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

    def infer(self, model: nn.Module, data: List[Any]) -> Union[torch.Tensor, np.ndarray, List[float]]:
        current_app.logger.info("In the inference function")
        return [0.1, 2.3]
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = self.transf(data).to(device)
        data = data.permute(1, 2, 0).view(-1, 1, 28, 28)
        out = model(data)

        preds = torch.argmax(out, axis=1)
        return preds.cpu().numpy()

    def init(self, model: nn.Module):
        current_app.logger.info("in the init function")
        def init_weights(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

        model.apply(init_weights)


class ExampleDataset(KubeDataset):

    def __init__(self, transform: transforms = None):
        super(ExampleDataset, self).__init__(dataset="mnist")
        self.transform = transform

    def __getitem__(self, index) -> T_co:
        x = self.data[index]
        y = self.labels[index]

        if self.transform:
            return self.transform(x), y.astype('int64')
        else:
            return x, y.astype('int64')

    def __len__(self):
        return len(self.data)


def main():
    net = Net()
    kubenet = KubeNet(net)
    return kubenet.start()
