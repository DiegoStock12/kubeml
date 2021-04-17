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

class MnistDataset(KubeDataset):

    def __init__(self):
        super().__init__("mnist")
        self.transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        return self.transf(x), y.astype('int64')

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

The other main component is the model class. This abstract class abstracts the complexity of distributing the training
among multiple workers, nodes and GPUs. The constructor only takes a torch model and the dataset as a parameter. The user only needs
to implement the abstract methods of the class, `train`, `infer`, `validate` `init` and `configure_optimizers` with the behavior they
want from the network.

The Kubenet exposes the `batch_size` and `lr` arguments which the user can change when starting the train job


```python
from kubeml import KubeModel
import torch
import torch.nn as nn
import numpy as np

class KubeLeNet(KubeModel):

    def __init__(self, network, dataset):
        super().__init__(network, dataset, gpu=True)
    
    @abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        pass

    # Train trains the model for an epoch and returns the loss
    @abstractmethod
    def train(self, x, y, batch_index) -> float:
        pass
    
    # Validate validates the model on the test data and returns a tuple
    # of (accuracy, loss)
    @abstractmethod
    def validate(self, x, y, batch_index) -> Tuple[float, float]:
        pass
    
    # Infer receives the data points or images as a list and returns 
    # the predictions of the network
    @abstractmethod
    def infer(self, data: List[Any]) -> Union[torch.Tensor, np.ndarray, List[float]]:
        pass
    
    # Init initializes the model in a particular way
    @abstractmethod
    def init(self, model: nn.Module):
       pass

```

An example implementation of the `init` and `train` functions can be done as follows

```python
    # Train trains the model for an epoch and returns the loss
     def train(self, x, y, batch_index) -> float:
        # define the device for training and load the data
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0
    
        self.optimizer.zero_grad()
        output = self(x)
    
        # compute loss and backprop
        # logging.debug(f'Shape of the output is {output.shape}, y is {y.shape}')
        loss = loss_fn(output, y)
        loss.backward()
    
        # step with the optimizer
        self.optimizer.step()
        total_loss += loss.item()
    
        if batch_index % 10 == 0:
            logging.info(f"Index {batch_index}, error: {loss.item()}")
    
        return total_loss
    
    # Intialize the network as a pytorch model
    def init(self, model):
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
    lenet = LeNet()
    dataset = MnistDataset()
    kubenet = KubeLeNet(lenet, dataset)
    return kubenet.start()
```

