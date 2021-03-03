#!/usr/bin/env bash

#
# build_python_lib.sh builds the kubeml library and publishes it to pypi
#

# directory where everything is saved
KUBEML_PY_DIR="/mnt/c/Users/diego/CS/thesis/python/kubeml"

echo "clearing previous binaries..."
rm -rf ${KUBEML_PY_DIR}/dist/ 2>&1

echo "building new binaries..."
python ${KUBEML_PY_DIR}/setup.py sdist bdist_wheel 2>&1


echo "check distribution..."
twine check




