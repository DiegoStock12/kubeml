package controller

import (
	"fmt"
	"github.com/gorilla/mux"
	"go.uber.org/zap"
	"net/http"
)

// Handle Kubernetes heartbeats
func (c *Controller) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

// Returns the functions used to handle requests
func (c *Controller) getHandler() http.Handler {
	r := mux.NewRouter()

	// training and inference
	r.HandleFunc("/train", c.train).Methods("POST")
	r.HandleFunc("/infer", c.infer).Methods("POST")

	// dataset proxy and methods
	r.HandleFunc("/dataset/{name}", c.getDataset).Methods("GET")
	r.HandleFunc("/dataset/{name}", c.storageServiceProxy).Methods("POST", "DELETE")
	r.HandleFunc("/dataset", c.listDatasets).Methods("GET")

	// get current tasks
	r.HandleFunc("/tasks", c.listTasks).Methods("GET")

	// history
	r.HandleFunc("/history/{taskId}", c.getHistory).Methods("GET")
	r.HandleFunc("/history/{taskId}", c.deleteHistory).Methods("DELETE")
	r.HandleFunc("/history", c.listHistories).Methods("GET")

	// k8s health handler
	r.HandleFunc("/health", c.handleHealth).Methods("GET")

	return r
}

// Starts the Controller API to handle requests
func (c *Controller) Serve(port int) {
	c.logger.Info("Starting controller API", zap.Int("port", port))
	addr := fmt.Sprintf(":%v", port)

	// start the server
	err := http.ListenAndServe(addr, c.getHandler())
	c.logger.Fatal("Controller quit", zap.Error(err))
}
