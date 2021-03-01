package main

import (
	"fmt"
	"github.com/fission/fission/pkg/crd"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"net/url"
	"strings"
)

const KubeMLNamespace = "kubeml"

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
	ip := svc.Status.LoadBalancer.Ingress[0].IP
	return ip
}



func main() {

	// with the config we can get the server address
	config, kubeClient, _, err := crd.GetKubernetesClient()
	if err != nil {
		panic(err)
	}

	// check if the load balancer ip is set
	fmt.Println(strings.Split(config.Host, ":"))

	// get list of services
	svc, err := kubeClient.CoreV1().Services(KubeMLNamespace).Get("controller", metav1.GetOptions{})
	if err != nil {
		panic(err)
	}

	// if there is a load balancer return the url of the load
	// balancer and the external port
	var address string
	var port int32
	if isLoadBalanced(svc) {
		address = getLoadBalancerIP(svc)
		port = svc.Spec.Ports[0].Port
	} else {
		u, err :=  url.Parse(config.Host)
		if err != nil {
			panic(err)
		}
		address = strings.Split(u.Host, ":")[0]
		port = svc.Spec.Ports[0].NodePort
	}

	cUrl, err := url.Parse(fmt.Sprintf("http://%v:%v", address, port))
	if err != nil {
		panic(err)
	}
	fmt.Println("URL is ", cUrl)


}
