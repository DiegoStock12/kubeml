#!/usr/bin/env bash

KUBEML_HOME="/mnt/c/Users/diego/CS/thesis/ml/pkg/kubeml-cli/kubeml"
CIFAR10_DATASET="/mnt/c/Users/diego/CS/thesis/ml/experiments/datasets/cifar10"

echo "Uploading CIFAR10 dataset"

"$KUBEML_HOME" dataset create --name cifar10 --traindata ${CIFAR10_DATASET}/cifar10_x_train.npy \
                                        --trainlabels ${CIFAR10_DATASET}/cifar10_y_train.npy \
                                        --testdata ${CIFAR10_DATASET}/cifar10_x_test.npy \
                                        --testlabels ${CIFAR10_DATASET}/cifar10_y_test.npy