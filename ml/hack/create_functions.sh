#!/usr/bin/env bash

KUBEML_HOME="/mnt/c/Users/diego/CS/thesis/ml/pkg/kubeml-cli/kubeml"
FUNC_HOME="/mnt/c/Users/diego/CS/thesis/ml/experiments/kubeml"

echo "creating lenet"
"$KUBEML_HOME" fn create --name lenet --code ${FUNC_HOME}/function_lenet.py

echo "creating resnet"
"$KUBEML_HOME" fn create --name resnet --code ${FUNC_HOME}/function_resnet34.py

echo "creating vgg"
"$KUBEML_HOME" fn create --name vgg --code ${FUNC_HOME}/function_vgg11.py