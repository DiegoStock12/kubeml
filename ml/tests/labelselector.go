package main

import (
	"fmt"
	"github.com/fission/fission/pkg/crd"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"strings"
)

const KubeMlNamespace = "kubeml"

func main() {

	var response string
	fmt.Print("This will delete all finished tasks, continue? (y/N) ")
	fmt.Scanf("%s", &response)

	switch strings.ToLower(response) {
	case "y":
		fmt.Println("Deleting finished tasks...")
	default:
		fmt.Println("Cancelling...")
		return
	}

	_, kubeClient, _, err := crd.GetKubernetesClient()
	if err != nil {
		panic(err)
	}

	opt := metav1.ListOptions{
		LabelSelector: "svc=controller",
	}

	fmt.Println(opt)

	svcs, err := kubeClient.CoreV1().Services(KubeMlNamespace).List(opt)
	if err != nil {
		panic(err)
	}

	for _, svc := range svcs.Items {
		fmt.Println(svc.Name)
	}

}
