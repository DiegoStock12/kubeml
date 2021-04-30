# KubeML


![KubeML CI](https://github.com/diegostock12/kubeml/actions/workflows/kubeml.yml/badge.svg?branch=master)


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

For installation, you need to add a few repositories to your helm repo list and install them. If you already have the repos
indexed in your local machine, you can directly run the `cluster_config.sh` script under `ml/hack` that will take care
of everything for you.

### Install Fission
KubeML requires a Kubernetes Cluster and Fission installed to work. To install fission,
it is recommended to use Helm.

```bash
$ export FISSION_NAMESPACE="fission"
$ kubectl create namespace $FISSION_NAMESPACE


$ helm repo add fission-charts https://fission.github.io/fission-charts/
$ helm repo update

# Install fission disabling custom prometheus
$ helm install -n fission fission fission-charts/fission-core --set prometheus.enabled=False
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
$ helm repo add stable https://charts.helm.sh/stable
$ helm repo update
$ helm install kubeml-metrics --namespace monitoring prometheus-community/kube-prometheus-stack \
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
    https://github.com/diegostock12/kubeml/releases/download/0.1.2/kubeml-0.1.2.tgz
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

Lastly, use the same train and validation methods as locally, simply referencing the KubeML Dataset. To define the distributed
training code we opt for a similar approach to what PyTorch Lighting does, structuring the code more than traditional pytorch.

The user does not need to call `.cuda()`, or `.to()` nor worry about `torch.no_grad()` or `model.eval()` and `model.train()`
and creating DataLoaders. The train and validation methods just have to be completed with the code for an iteration and return
the loss, KubeML takes care of everything else.

```python
from kubeml import KubeModel
import torch
import torch.nn as nn
import numpy as np

class KubeLeNet(KubeModel):

    def __init__(self, network: nn.Module, dataset: MnistDataset):
        # notice that if you turn gpu on and there is no GPU on the machine
        # the initialization would fail
        super().__init__(network, dataset, gpu=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        sgd = SGD(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return sgd

    def init(self, model: nn.Module):
        def init_weights(m: nn.Module):
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.01)
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.01)
        
        model.apply(init_weights)

    def train(self, batch, batch_index) -> float:

        x, y = batch

        self.optimizer.zero_grad()
        output = self(x)
        loss = F.cross_entropy(output, y)
        loss.backward()

        self.optimizer.step()

        if batch_index % 10 == 0:
            self.logger.info(f"Index {batch_index}, error: {loss.item()}")

        return loss.item()

    def validate(self, batch, batch_index) -> Tuple[float, float]:
        
        x, y = batch

        # forward pass and loss accuracy calculation
        output = self(x)
        loss = F.cross_entropy(output, y)  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(y.view_as(pred)).sum().item()

        accuracy = 100. * correct / self.batch_size
        self.logger.debug(f'accuracy {accuracy}')

        return accuracy, loss.item()

    def infer(self, data: List[Any]) -> Union[torch.Tensor, np.ndarray, List[float]]:
        pass
    
```

### Define the Function Entrypoint

In the main function, create the network object and start the function

```python
def main():
    lenet = LeNet()
    dataset = MnistDataset()
    kubenet = KubeLeNet(lenet, dataset)
    return kubenet.start()
```

## Training a Network

First, you need to download the kubeml client. This can be found in the downloads section of the releases, or can be downloaded
like:

```bash
$ curl -Lo kubeml https://github.com/diegostock12/kubeml/releases/download/0.1.2/kubeml \
&& chmod +x kubeml
```

### Deploying a Function

Once you have written you function code, you can deploy it using the KubeML CLI.

```bash
$ kubeml function create --name example --code network.py
```

If you want to update the code of the function after fixing issues, the best way to do so is 
to use the fission CLI. This functionality might be added in the future to the kubeML CLI.

```bash
$ fission function update --name example --code network.py
```

Would update the function code.

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

Other options include setting the parallelism `--parallelism`, `--static`, which keeps the parallelism stable (recommended for testing)
, and `validate-every` which sets the number of epochs between validations.

### Testing Locally

To test in your computer some options tested are MiniKube or MicroK8s. MicroK8s makes it easier to turn on GPU suppost
by doing `microk8s enable gpu`. The only thing needed are NVIDIA Drivers build for CUDA 10.1+. 
