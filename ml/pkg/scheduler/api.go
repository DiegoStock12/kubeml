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

// newParallelism listens to the TrainJobs of the Parameter Server and their
// requests for a new level of parallelism.
// To make this doable we need to put the request in a queue and wait for the scheduler
// to get it and schedule it
func (s *Scheduler) newParallelism(w http.ResponseWriter, r *http.Request) {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		s.logger.Error("Failed to get the training request from the TrainJob", zap.Error(err))
		http.Error(w, "Failed to read request", http.StatusInternalServerError)
		return
	}

	// Receive the train task from the job
	// and insert it in the queue to be scheduled
	var task api.TrainTask

	// read the train request
	err = json.Unmarshal(body, &task)
	if err != nil {
		s.logger.Error("Failed to parse the trainjob request",
			zap.Error(err),
			zap.String("payload", string(body)))
		http.Error(w, "Failed to decode the request", http.StatusInternalServerError)
		return
	}

	s.logger.Debug("Received request for new parallelism",
		zap.Any("task", task))

	// Add the request to scheduler queue
	s.queue.pushTask(&task)

	w.WriteHeader(http.StatusOK)

}

// train receives the TrainRequest from the controller and enqueues them
func (s *Scheduler) train(w http.ResponseWriter, r *http.Request) {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		s.logger.Error("Failed to get the training request from the API", zap.Error(err))
		http.Error(w, "Failed to read request", http.StatusInternalServerError)
		return
	}

	var req api.TrainRequest

	// read the train request
	err = json.Unmarshal(body, &req)
	if err != nil {
		s.logger.Error("Failed to parse the train request",
			zap.Error(err),
			zap.String("payload", string(body)))
		http.Error(w, "Failed to decode the request", http.StatusInternalServerError)
		return
	}

	// Create the jobId and push to queue
	id := createJobId()

	// TODO now add it directly to the task queue
	t := api.TrainTask{
		Parameters:  req,
		Parallelism: -1,
		JobId:       id,
		ElapsedTime: -1,
	}

	s.logger.Debug("Adding task to queue",
		zap.Any("task", t))
	s.queue.pushTask(&t)

	s.logger.Debug("here")

	w.WriteHeader(http.StatusOK)
	_, err = w.Write([]byte(id))
	if err != nil{
		s.logger.Error("error writing response", zap.Error(err))
	}
}

// Handle requests to infer with some datapoints
// TODO unimplemented
func (s *Scheduler) infer(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

// Handle heartbeats from Kubernetes
func (s *Scheduler) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

// Create the handler for the scheduler to receive requests from the API
func (s *Scheduler) GetHandler() http.Handler {
	r := mux.NewRouter()
	r.HandleFunc("/job", s.newParallelism).Methods("POST")
	r.HandleFunc("/train", s.train).Methods("POST")
	r.HandleFunc("/infer", s.infer).Methods("POST")
	r.HandleFunc("/health", s.handleHealth).Methods("GET")
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
