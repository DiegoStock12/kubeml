# KubeML

KubeML, Serverless Neural Network Training on Kubernetes with transparent load distribution. 
Write code as if you were to run it locally, and easily deploy it to a distributed
training platform.

---

## Motivation

Training a neural network distributedly can be quite a complex task. Many times,
allocating a GPU cluster, changing the local code to use special methods and classes
is a necessary and costly step. Added to this, optimizing data distribution to processes
is left to the user.

With KubeML the goal is simple. Write code as you would to test locally, and once 
you are ready, write a python function and deploy it to KubeML. No configuration or yaml files,
almost no changes to the code, just a couple lines of code and a CLI command, and your 
network will be automatically distributed across the cluster.

## Description

KubeML runs your ML code on top of Kubernetes, using Fission as the serverless 
platform of choice. Once a function is deployed, you can upload datasets and start 
training jobs with just a single command.


## Table of Contents

* [Components](#components)
* [Installation](#installation)
    * [Install Fission](#install-fission)
    * [Install Prometheus](#install-prometheus)
    * [Install KubeML](#install-kubeml)
* [Writing a Function](#writing-a-function)
    * [Define the Dataset](#define-the-dataset-class)
    * [Define the Network](#define-the-network)
    * [Define the Function Entrypoint](#define-the-function-entrypoint)
* [Training a Network](#training-a-network)
    * [Deploy a Function](#deploying-a-function)
    * [Upload a dataset](#uploading-a-dataset)
    * [Start Training](#starting-the-training)
    

## Components

1. _Controller_: Exposes the KubeML resources to an API

2. _Scheduler_: Decides what tasks to schedule and decides the parallelism of functions. Functions
are scaled in or out during training dinamically, based on the current load and performance of the cluster.

3. _Parameter Server_: Starts and manages the training job pods, each of which will be responsible for a network.

4. _Train Job_: Each deployed in a standalone pod, will manage the reference model of a train task using a 
Parameter Server architecture. 

5. _Storage Service_: Processes datasets to efficiently store them so they can be automatically loaded by the functions.

6. _ML Functions_: Run on a custom python environment, and execute PyTorch code. The lifecycle of the network is abstracted 
away for the user. The functions automatically load the right minibatches from the storage service, and train the network 
using a data parallel approach.

## Installation

### Install Fission
KubeML requires a Kubernetes Cluster and Fission installed to work. To install fission,
it is recommended to use Helm.

```bash
$ export FISSION_NAMESPACE="fission"
$ kubectl create namespace $FISSION_NAMESPACE

# Install fission disabling custom prometheus
$ helm install --namespace $FISSION_NAMESPACE --name-template fission \
    https://github.com/fission/fission/releases/download/1.12.0/fission-core-1.12.0.tgz \
    --set prometheus.enabled=false
```

### Install Prometheus

KubeML exposes metrics to prometheus so you are able to track the process of training jobs, 
with metrics such as parallelism, accuracy, train or validation loss or epoch time. To install
prometheus with Helm:

First create the monitoring namespace

```bash
$ export METRICS_NAMESPACE=monitoring
$ kubectl create namespace $METRICS_NAMESPACE
```

Then get the helm chart and install

```bash
$ helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
$ helm repo add stable https://kubernetes-charts.storage.googleapis.com/
$ helm repo update
$ helm install fission-metrics --namespace monitoring prometheus-community/kube-prometheus-stack \
  --set kubelet.serviceMonitor.https=true \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues=false
```

### Install KubeML

The best way to install KubeML is to use Helm to install the provided charts. For installing in the preferred namespace,
you can use the following commands:

```bash
$ export KUBEML_NAMESPACE=kubeml
$ kubectl create namespace $KUBEML_NAMESPACE

# Install all the components in the kubeml namespace
$ helm install --namespace $KUBEML_NAMESPACE --name-template kubeml \
    https://github.com/diegostock12/kubeml/releases/download/0.1.0/kubeml-0.1.0.tgz
```

## Writing a Function

KubeML supports writing function code in PyTorch. After you have written the local code, you only need to 
define a main method, which will be invoked in the serverless function, and define your train, validation and init 
methods in the network. 

Then, using the custom objects from the [KubeML Python module](https://pypi.org/project/kubeml/), start the training process.

### Define the Dataset Class

```python
from kubeml import KubeDataset
from torchvision import transforms

class MnistDataset(KubeDataset):
    
    def __init__(self, transform: transforms = None):
        # use the dataset name as uploaded to KubeML storage
        super(MnistDataset, self).__init__(dataset="mnist")
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

### Define the Network

Lastly, use the same train and validation methods as localy, simply referencing the KubeML Dataset.

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
        dataset = MnistDataset()
        train_loader = data.DataLoader(dataset, batch_size=self.args.batch_size)
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

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
        def init_weights(m: nn.Module):
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.constant_(m.bias, 0.01)
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.constant_(m.bias, 0.01)
        
        model.apply(init_weights)
```

### Define the Function Entrypoint

In the main function, create the network object and start the function

```python
def main():
    # Create the PyTorch Model
    net = Net()
    # create the KubeML model with the network as parameter
    kubenet = KubeNet(net)
    return kubenet.start()
```

## Training a Network

### Deploying a Function

Once you have written you function code, you can deploy it using the KubeML CLI.

```bash
$ kubeml function create --name example --code network.py
```

### Uploading a Dataset

To upload a dataset, create four different files (.npy or .pkl formats are accepted), with the train features and labels, and the test features and labels.
After that, you can easily upload it with the CLI.

```bash
$ kubeml dataset create --name mnist \
    --traindata train.npy \
    --trainlabels y_train.npy \
    --testdata test.npy \
    --testlabels y_test.npy
```

### Starting the Training

After the dataset and functions are created, start the training using the network and dataset names defined above.

```bash
$ kubeml train \
    --function example \
    --dataset mnist \
    --epochs 10 \
    --batch 128 \
    --lr 0.01
```
