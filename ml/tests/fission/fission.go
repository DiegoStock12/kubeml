package main

import (
	"fmt"
	"github.com/fission/fission/pkg/crd"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"os"
)

func main() {

	_ = os.Setenv("KUBECONFIG", "C:\\Users\\diego\\.kube\\config")

	fissionClient, _, _, err := crd.MakeFissionClient()
	if err != nil {
		panic(err)
	}

	functions, err := fissionClient.CoreV1().Functions("").List(metav1.ListOptions{})
	if err != nil {
		panic(err)
	}
	for _, fun := range functions.Items {
		fmt.Println(fun.Name, fun.Spec.Environment.Name , fun.CreationTimestamp.Time)
	}

}
