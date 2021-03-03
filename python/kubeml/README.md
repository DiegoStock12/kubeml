# KubeML

KubeML provides wrappers and tools that allow the interaction with user code written in PyTorch
with the distributed training and serving functionality offered by KubeML

## Installing

Install and update using pip

```text
pip install kubeml
```

## Usage

The main functionality offered is in the shape of Models and Datasets. A KubeDataset is a convenience wrapper over a
torch dataset which, like when using torch, users extend with their own functionality to adapt to their data. A simple
example of how to create a dataset to train with KubeML is seen below.

### The Dataset class

```python
from kubeml import KubeDataset
from torchvision import transforms

class ExampleDataset(KubeDataset):

    def __init__(self, transform: transforms = None):
        super(ExampleDataset, self).__init__(dataset="mnist")
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        if self.transform:
            return self.transform(x), y.astype('int64')
        else:
            return x, y.astype('int64')

    def __len__(self):
        return len(self.data)
```

The user only needs to provide to the constructor the name of the dataset as was uploaded to the KubeML storage, 
the dataset will take care of fetching only the corresponding minibatches of data so that the network can be trained
with a model parallel approach.

As with a normal torch dataset, the user must implement the `__getitem__` and `__len__` methods to iterate over the dataset.
The dataset exposes two member variables:
1. `data` Holds the features used as input to the network
2. `labels` Holds the output labels

Both are saved as numpy arrays.

### The Model class

The other main component is the model class. This class abstracts the complexity of distributing the training
among multiple workers, nodes and GPUs. The constructor only takes a torch model as a parameter. The user only needs
to implement the unimplemented methods of the class, `train`, `infer`, `validate` and `init` with the behavior they
want from the network.

The Kubenet exposes the `args` member variable which holds the arguments used by KubeML such
as batch size, learning rate... chosen by the user, which can be accessed from the training methods.


```python
from kubeml import KubeModel
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch import optim
from torchvision import transforms

class KubeNet(KubeModel):

    def __init__(self, network: nn.Module):
        super().__init__(network)

    # Train trains the model for an epoch and returns the loss
    def train(self, model: nn.Module) -> float:
        raise NotImplementedError
    
    # Validate validates the model on the test data and returns a tuple
    # of (accuracy, loss)
    def validate(self, model: nn.Module) -> Tuple[float, float]:
        raise NotImplementedError
    
    # Infer receives the data points or images as a list and returns 
    # the predictions of the network
    def infer(self, model: nn.Module, data: List[Any]) -> Union[torch.Tensor, np.ndarray, List[float]]:
        raise NotImplementedError
    
    # Init initializes the model in a particular way
    def init(self, model: nn.Module):
        raise NotImplementedError

```

An example implementation of the `init` and `train` functions can be done as follows

```python

    # Train trains the model for an epoch and returns the loss
    def train(self, model: nn.Module) -> float:
        
        # Set the device as GPU, create the Example KubeDataset
        # and use a torch dataloader with the 
        device = torch.device("cuda")
        dataset = ExampleDataset()
        train_loader = data.DataLoader(dataset, batch_size=self.args.batch_size)
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

            optimizer.step()

        return loss.item()
    
    # Intialize the network as a pytorch model
    def init(self, model: nn.Module):
        def init_weights(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

        model.apply(init_weights)
```

## Writing the training function

At the moment of creating a serverless function which will serve as a worker for the model training process, the 
steps are simple, simply write the code initializing the network in the `main` method of the function, and call
`start` on the KubeML model.

```python
def main():
    # Create the PyTorch Model
    net = Net()
    # create the KubeML model with the network as parameter
    kubenet = KubeNet(net)
    return kubenet.start()
```

