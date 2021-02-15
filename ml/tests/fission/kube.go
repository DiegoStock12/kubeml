package main

import (
	"container/list"
	"fmt"
	"github.com/fission/fission/pkg/crd"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"math/rand"
	"os"
	"sync"
	"time"
)

type SyncList struct {
	*list.List
	sync.Mutex
}

func NewSyncList() *SyncList {

	return &SyncList{
		List:  list.New(),
		Mutex: sync.Mutex{},
	}
}

func worker(s *SyncList, wg *sync.WaitGroup) {
	fmt.Println("worker starting...")
	for i := 0; i < 100; i++ {
		s.Lock()
		s.PushBack(rand.Intn(100))
		s.Unlock()
	}

	wg.Done()
}


func getPod() *corev1.Pod{

	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-busypod",
			Namespace: "default",
			Labels: map[string]string{
				"app": "test",
			},

		},
		Spec:       corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name: "busybox",
					Image: "busybox:1.28",
					ImagePullPolicy: corev1.PullIfNotPresent,
					Command: []string{
						"sleep",
						"300",
					},
					Ports: []corev1.ContainerPort{
						{
							Name:          "ssh-busybox",
							HostPort:      0,
							ContainerPort: 22,
							Protocol:      "TCP",
												},
					},

				},
			},
		},
	}


}

func getJobPod(client *kubernetes.Clientset)  {
	
	pod, err := client.CoreV1().Pods("kubeml").List(metav1.ListOptions{})
	if err != nil {panic(err)}
	for _, p := range pod.Items{
		fmt.Println(p.Name, p.Status.PodIP, p.Status.Phase)
	}
	
}

func main() {

	// The make fission client and so on needs the KUBECONFIG variable to be set
	_ = os.Setenv("KUBECONFIG", "C:\\Users\\diego\\.kube\\config")



	//// Access the kubernetes API directly from the code
	//config, _ := clientcmd.BuildConfigFromFlags("", "C:\\Users\\diego\\.kube\\config")
	//clienset, _ := kubernetes.NewForConfig(config)
	//svcs, _ := clienset.CoreV1().Pods("fission-function").List(metav1.ListOptions{})
	//for _, i := range svcs.Items {
	//	fmt.Println(i.Name)
	//}


	// Create the multiple clients from the fission stuff
	_, kubeClient, _, err := crd.MakeFissionClient()
	if err != nil {
		panic(err)
	}


	getJobPod(kubeClient)
	os.Exit(0)


	pods, err := kubeClient.CoreV1().Pods("kubeml").List(metav1.ListOptions{})
	if err != nil {panic(err)}

	for _, p := range pods.Items {
		fmt.Println(p.Spec.Containers[0].ReadinessProbe)
	}

	// try to create and delete a pod
	busyPod := getPod()
	podref, err := kubeClient.CoreV1().Pods(busyPod.Namespace).Create(busyPod)
	if err != nil {panic(err)}

	fmt.Println("Created pod", podref.Name)

	// list the pods in the namespace
	pods, err = kubeClient.CoreV1().Pods(busyPod.Namespace).List(metav1.ListOptions{})
	if err != nil {panic(err)}

	for _, p := range pods.Items{
		fmt.Println(p.Name, p.Spec.Containers, p.Spec.Containers[0].Ports)
	}


	time.Sleep(30 * time.Second)
	fmt.Println("Deleting pod...")
	// delete the busybox pod
	err = kubeClient.CoreV1().Pods(podref.Namespace).Delete(podref.Name, &metav1.DeleteOptions{})
	if err != nil {panic(err)}

	//
	//env, _ := fissionClient.CoreV1().Functions("").List(metav1.ListOptions{})
	//for _, e := range env.Items {
	//	fmt.Println(e.Name)
	//}

}
