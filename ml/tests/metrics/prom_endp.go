package main

import (
	"fmt"
	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"io/ioutil"
	"math/rand"
	"strconv"

	"log"
	"net/http"
	"time"
)

var (
	operationsProcessed = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "kube_processed_total",
			Help: "The total number of requests processed",
		})

	id = 0
)

func recordMetrics() {
	// Simply increment the counter every couple of seconds
	go func() {
		for {
			operationsProcessed.Inc()
			time.Sleep(5 * time.Second)
		}
	}()
}

func unregister(w http.ResponseWriter, r *http.Request) {
	log.Println("Deregistering metric")
	prometheus.Unregister(operationsProcessed)
	log.Println("unregistered...")
}

// job creates a new job that runs for a while and shows prometheus metrics
func job(w http.ResponseWriter, r *http.Request) {
	go func() {
		myId := id
		id++
		log.Println("Started job with id", myId)

		// create the prometheus metric
		loss := promauto.NewGauge(prometheus.GaugeOpts{
			Name: fmt.Sprintf("kubeml_job_%d_loss", myId),
			Help: fmt.Sprintf("Loss values for job %d", myId),
		})
		defer prometheus.Unregister(loss)

		for i := 0; i < 20; i++ {
			log.Println("Function", myId, "setting loss")
			loss.Set(rand.Float64())
			time.Sleep(3 * time.Second)
		}

		log.Println("Job", myId, "exiting...")
	}()
}

func finish(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := strconv.Atoi(vars["funcId"])
	if err != nil {
		http.Error(w, "error", http.StatusBadRequest)
		return
	}

	//time.Sleep(20 * time.Second)

	fmt.Println("Got response from function", id)
	w.Write([]byte("hola"))
	w.WriteHeader(http.StatusOK)

}

// test reads the request body if it is empty
func test(w http.ResponseWriter, r *http.Request) {
	body := r.Body
	if body == http.NoBody {
		fmt.Println("Body is nil")
	} else {
		received, err := ioutil.ReadAll(body)
		fmt.Println(len(received))
		if len(received) == 0 {
			fmt.Println("Received empty body")
		}
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("Received error", string(received))
	}

}

func main() {
	recordMetrics()

	r := mux.NewRouter()

	http.Handle("/metrics", promhttp.Handler())
	//r.HandleFunc("/unregister", unregister)
	//r.HandleFunc("/job", job)
	//r.HandleFunc("/test", test)
	r.HandleFunc("/finish/{funcId}", finish).Methods("POST")
	log.Fatal(http.ListenAndServe(":9999", r))
}
