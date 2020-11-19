package scheduler

import (
	"encoding/json"
	"fmt"
	"github.com/gorilla/mux"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"

	"github.com/diegostock12/thesis/ml/pkg/api"
)

// Api exposed by the scheduler to interact with the API server

// Handle requests from the API to infer tasks
func (s *Scheduler) scheduleTrainTask(w http.ResponseWriter, r *http.Request)  {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		s.logger.Error("Failed to get the training request from the API", zap.Error(err))
		http.Error(w, "Failed to read request", http.StatusInternalServerError)
		return
	}

	req := api.TrainRequest{}
	// read the train request
	err = json.Unmarshal(body, &req)
	if err != nil {
		s.logger.Error("Failed to parse the train request",
			zap.Error(err),
			zap.String("payload", string(body)))
		http.Error(w, "Failed to decode the request", http.StatusInternalServerError)
		return
	}

	// Add the request to the channel of the scheduler
	// TODO see how to handle this so we just answer when the task is actually scheduled
	// TODO send alongside the request a response channel and get the parameters chosen by the scheduler
	s.schedChan <- &req

	w.WriteHeader(http.StatusOK)
}


// Handle requests to infer with some datapoints
// TODO unimplemented
func (s *Scheduler) scheduleInferenceTask(w http.ResponseWriter, r *http.Request)  {
	w.WriteHeader(http.StatusOK)
}

// Handle heartbeats from Kubernetes
func (s *Scheduler) healthHandler(w http.ResponseWriter, r *http.Request)  {
	w.WriteHeader(http.StatusOK)
}


// Create the handler for the scheduler to receive requests from the API
func (s *Scheduler) GetHandler() http.Handler {
	r := mux.NewRouter()
	r.HandleFunc("/scheduleTrainTask", s.scheduleTrainTask).Methods("POST")
	r.HandleFunc("/scheduleInferenceTask", s.scheduleInferenceTask).Methods("POST")
	r.HandleFunc("/health", s.healthHandler).Methods("GET")
	return r
}

// Expose the API
func (s *Scheduler) Serve(port int) {
	s.logger.Info("Starting scheduler api", zap.Int("port", port))
	addr := fmt.Sprintf(":%v", port)

	// Start serving the endpoint
	err := http.ListenAndServe(addr, s.GetHandler())
	s.logger.Fatal("Scheduler API done", zap.Error(err))

}

