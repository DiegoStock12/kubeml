package client

import (
"fmt"
"github.com/fission/fission/pkg/crd"
"github.com/pkg/errors"
corev1 "k8s.io/api/core/v1"
metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
"net/url"
"strings"
)

const KubeMLNamespace = "kubeml"

// getControllerUrl extracts the controller url from the kubeconfig file
// and the service definition in the kubeml namespace in the cluster
func getControllerUrl() (string, error) {

	// with the config we can get the server address
	config, kubeClient, _, err := crd.GetKubernetesClient()
	if err != nil {
		return "", errors.Wrap(err, "error creating kubernetes client")
	}

	// get list of services
	svc, err := kubeClient.CoreV1().Services(KubeMLNamespace).Get("controller", metav1.GetOptions{})
	if err != nil {
		return "", errors.Wrap(err, "error finding controller service")
	}

	// if there is a load balancer return the url of the load
	// balancer and the external port
	var address string
	var port int32
	if isLoadBalanced(svc) {
		address = getLoadBalancerIP(svc)
		port = svc.Spec.Ports[0].Port
	} else {
		u, err := url.Parse(config.Host)
		if err != nil {
			return "", errors.Wrap(err, "error parsing server url")
		}
		address = strings.Split(u.Host, ":")[0]
		port = svc.Spec.Ports[0].NodePort
	}

	controllerUrl := fmt.Sprintf("http://%v:%v", address, port)

	return controllerUrl, nil

}

// isLoadBalanced returns whether the service has an active load balancer
func isLoadBalanced(svc *corev1.Service) bool {
	ingresses := svc.Status.LoadBalancer.Ingress
	if len(ingresses) == 0 {
		return false
	}
	return true
}

// getLoadBalancerIP returns the IP associated to the load balancer
func getLoadBalancerIP(svc *corev1.Service) string {
	return svc.Status.LoadBalancer.Ingress[0].IP
}

