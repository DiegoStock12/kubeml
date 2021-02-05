package main

import (
	"fmt"
	"github.com/fission/fission/pkg/crd"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func main() {
	fissionClient, _, _, err := crd.MakeFissionClient()
	if err != nil {
		panic(err)
	}

	functions, err :=fissionClient.CoreV1().Functions("").List(metav1.ListOptions{})
	if err != nil {
		panic(err)
	}
	for _, fun := range functions.Items {
		fmt.Println(fun)
	}
}
