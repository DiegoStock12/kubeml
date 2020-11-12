package main

import (
	"container/list"
	"fmt"
	"github.com/fission/fission/pkg/crd"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"math/rand"
	"os"
	"sync"
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

func main() {

	// The make fission client and so on needs the KUBECONFIG variable to be set
	_ = os.Setenv("KUBECONFIG", "C:\\Users\\diego\\.kube\\config")

	// Access the kubernetes API directly from the code
	config, _ := clientcmd.BuildConfigFromFlags("", "C:\\Users\\diego\\.kube\\config")
	clienset, _ := kubernetes.NewForConfig(config)
	svcs, _ := clienset.CoreV1().Pods("fission-function").List(metav1.ListOptions{})
	for _, i := range svcs.Items {
		fmt.Println(i.Name)
	}

	// Create the multiple clients from the fission stuff
	fissionClient, _, _, err := crd.MakeFissionClient()
	if err != nil {
		panic(err)
	}

	fun := fissionClient.CoreV1().Functions("fission-function")
	f, _ := fun.Get("example", metav1.GetOptions{})

	fmt.Println(f.Kind)

	//
	//_, err := g.Client().V1().Function().List("fission-function")
	//if err != nil {
	//	panic(err)
	//}

	//var wg sync.WaitGroup
	//s := NewSyncList()

	//s := NewSyncList()
	//s.Lock()
	//s.PushBack(12)
	//s.PushBack(10)
	//s.Unlock()

	//for i := 0 ; i < 100 ; i++{
	//	wg.Add(1)
	//	go worker(s, &wg)
	//}
	//
	//wg.Wait()

	//fmt.Println("Length of the queue is ", s.Len())

	//for e:= s.Front(); e!= nil ; e = e.Next() {
	//	fmt.Println(e.Value)
	//}

}
