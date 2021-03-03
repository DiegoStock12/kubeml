package main

import (
	"fmt"
	"github.com/fission/fission/pkg/crd"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
)

// returns another service for the controller
func getService() *corev1.Service {

	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "controller-service-2",
			Namespace: "kubeml",
			Labels:    map[string]string{"svc": "controller"},
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{"svc": "controller"},
			Type:     corev1.ServiceTypeClusterIP,
			Ports: []corev1.ServicePort{
				{
					Port:       80,
					TargetPort: intstr.FromInt(9090),
				},
			},
		},
	}

	return svc
}

func deleteService(client *kubernetes.Clientset, name string) error {
	err := client.CoreV1().Services("kubeml").Delete(name, &metav1.DeleteOptions{})
	return err
}

func main() {

	_, kubeClient, _, err := crd.GetKubernetesClient()
	if err != nil {
		panic(err)
	}

	//err = deleteService(kubeClient, "controller-service-2")
	//if err != nil {
	//	panic(err)
	//}

	//get a list of services (the controller service)
	svc, err := kubeClient.CoreV1().Services("kubeml").Get("controller", metav1.GetOptions{})
	if err != nil {
		panic(err)
	}

	fmt.Println(svc.Status)

	//// create
	//svc := getService()
	//svc, err = kubeClient.CoreV1().Services("kubeml").Create(svc)
	//if err != nil {
	//	panic(err)
	//}

}
