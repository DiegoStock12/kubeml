package main

import (
	"fmt"
	"github.com/fission/fission/pkg/crd"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"net/url"
)

const KubeMLNamespace = "kubeml"

func main() {

	// with the config we can get the server address
	config, kubeClient, _, err := crd.GetKubernetesClient()
	if err != nil {
		panic(err)
	}

	fmt.Println(config.Host)
	hostUrl, err := url.Parse(config.Host)
	if err != nil {panic(err)}

	// need to separate the port from the host and add the port from the controller instead
	// so we can access it
	fmt.Println(hostUrl, hostUrl.Port(), hostUrl.Host)


	// get list of services
	svc, err := kubeClient.CoreV1().Services(KubeMLNamespace).Get("controller", metav1.GetOptions{})
	if err != nil {
		panic(err)
	}

	fmt.Println(svc.Name, svc.Spec.Ports[0].NodePort)

}
