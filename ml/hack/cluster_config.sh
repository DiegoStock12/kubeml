#!/bin/bash

#
# cluster_config.sh - Automate setup of cluster
#


# declare the namespaces
FISSION_NAMESPACE="fission"
MONITORING_NAMESPACE="monitoring"
KUBEML_NAMESPACE="kubeml"

# check if kubectl is installed
if ! command -v kubectl >/dev/null 2>&1; then
    echo "kubectl is not installed"
    exit 1;
fi

# Check if helm is installed
if ! command -v helm >/dev/null 2>&1 ; then
    echo "helm is not installed."
    exit 1;
fi

# Create the fission release
echo "Deploying fission..."

kubectl create namespace $FISSION_NAMESPACE
helm install --namespace $FISSION_NAMESPACE --name-template fission \
    https://github.com/fission/fission/releases/download/1.12.0/fission-core-1.12.0.tgz \
    --set prometheus.enabled=false \
    2>&1

echo "Fission deployed!"


# if the env variable is not set create monitoring namespace and resources
if [[ -z $MONITORING ]]; then
  echo "Deploying prometheus..."

  kubectl create namespace $MONITORING_NAMESPACE
  helm install kubeml-metrics --namespace $MONITORING_NAMESPACE prometheus-community/kube-prometheus-stack \
  --set kubelet.serviceMonitor.https=true \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues=false \
  2>&1

  echo "Prometheus deployed!"
fi

# Deploy the kubeml charts
echo "Deploying kubeml"

kubectl create namespace $KUBEML_NAMESPACE
helm install kubeml --namespace $KUBEML_NAMESPACE  \
    ../charts/kubeml-0.1.2.tgz \
#    https://github.com/diegostock12/kubeml/releases/download/0.1.2/kubeml-0.1.2.tgz \
    2>&1

echo "kubeml deployed!! all done"

