package main

import (
	"fmt"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"math/rand"

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



func main() {
	recordMetrics()


	http.Handle("/metrics", promhttp.Handler())
	http.HandleFunc("/unregister", unregister)
	http.HandleFunc("/job", job)
	log.Fatal(http.ListenAndServe(":8000", nil))
}
