#!/usr/bin/env bash

KUBEML_HOME="/mnt/c/Users/diego/CS/thesis/ml/pkg/kubeml-cli/kubeml"
CIFAR100_DATASET="/mnt/c/Users/diego/CS/thesis/ml/experiments/datasets/cifar100"

echo "Uploading CIFAR100 dataset"

"$KUBEML_HOME" dataset create --name cifar100 --traindata ${CIFAR100_DATASET}/cifar100_x_train.npy \
                                        --trainlabels ${CIFAR100_DATASET}/cifar100_y_train.npy \
                                        --testdata ${CIFAR100_DATASET}/cifar100_x_test.npy \
                                        --testlabels ${CIFAR100_DATASET}/cifar100_y_test.npy