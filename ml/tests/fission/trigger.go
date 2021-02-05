package main

import (
	"fmt"
	fv1 "github.com/fission/fission/pkg/apis/core/v1"
	"github.com/fission/fission/pkg/crd"
	"github.com/pkg/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"net/http"
	"os"
)

//const DEFAULT_NAMESPACE = "default"

func createTrigger(fissionClient *crd.FissionClient, name string, methods []string) error {

	for _, method := range methods {
		ht := &fv1.HTTPTrigger{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: DEFAULT_NAMESPACE,
			},
			Spec: fv1.HTTPTriggerSpec{
				RelativeURL: "/" + name,
				Method:      method,
				FunctionReference: fv1.FunctionReference{
					Type: fv1.FunctionReferenceTypeFunctionName,
					Name: name,
				},
				IngressConfig: fv1.IngressConfig{
					Annotations: nil,
					Path:        "/" + name,
					Host:        "*",
					TLS:         "",
				},
			},
		}

		_, err := fissionClient.CoreV1().HTTPTriggers(DEFAULT_NAMESPACE).Create(ht)
		if err != nil {
			return errors.Wrap(err, "unable to create http trigger")
		}

	}

	return nil

}

func main() {
	_ = os.Setenv("KUBECONFIG", "C:\\Users\\diego\\.kube\\config")


	fissionClient, _, _, err := crd.MakeFissionClient()
	if err != nil {
		panic(err)
	}

	fmt.Println("trying to create trigger")
	err = createTrigger(fissionClient, "test", []string{http.MethodGet})
	if err != nil {
		panic(err)
	}

	fmt.Println("Exiting...")

}
