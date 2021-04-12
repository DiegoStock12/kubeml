#!/bin/bash

KUBEML_HOME="../pkg/kubeml-cli/kubeml"
FUNC_HOME="../experiments/kubeml"

echo "creating lenet"
"$KUBEML_HOME" fn create --name lenet --code ${FUNC_HOME}/function_lenet.py

echo "creating resnet"
"$KUBEML_HOME" fn create --name resnet --code ${FUNC_HOME}/function_resnet34.py

echo "creating vgg"
"$KUBEML_HOME" fn create --name vgg --code ${FUNC_HOME}/function_vgg11.py