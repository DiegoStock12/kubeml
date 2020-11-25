package controller

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/gorilla/mux"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"
)


const (
	schedulerTrainPath = "/scheduleTrainTask"
	schedulerInferencePath = "/scheduleInferenceTask"

	// TODO change this is we use another namespace
	schedulerURL = "http://scheduler.default"
)

// Task types to send to the scheduler
const(
	TrainTask TaskType = iota
	InferenceTask
)

type TaskType int

// Send a post request to the scheduler endpoints
func (c *Controller) sendRequestToScheduler(req interface{}, task TaskType) {

	switch task {
	case TrainTask:
		// build the url
		url := schedulerURL + schedulerTrainPath
		c.logger.Debug("Sending train request to scheduler at", zap.String("url", url))
		// Create the request body
		reqBody, err := json.Marshal(req.(api.TrainRequest))
		if err != nil {
			c.logger.Error("Could not marshall the request",
				zap.Any("body", req),
				zap.Error(err))
		}
		_, err = http.Post(url, "application/json", bytes.NewBuffer(reqBody))
		if err != nil {
			c.logger.Error("Error sending request",
				zap.Error(err))
		}

	case InferenceTask:
		url := schedulerURL + schedulerInferencePath
		c.logger.Debug("Sending inference request to scheduler at", zap.String("url", url))
		// Create the request body
		reqBody, err := json.Marshal(req.(api.InferRequest))
		if err != nil {
			c.logger.Error("Could not marshall the request",
				zap.Any("body", req),
				zap.Error(err))
		}
		_, err = http.Post(url, "application/json", bytes.NewBuffer(reqBody))
		if err != nil {
			c.logger.Error("Error sending request",
				zap.Error(err))
		}
	
	}
}

// TODO this should generate a train ID similar to pods (resnet-uid) that could be used to acces the results layer
// TODO this could be related to the ID of the parameter server
// Handle a train request and forward it to the scheduler
func (c *Controller) handleTrainRequest(w http.ResponseWriter, r *http.Request)  {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		c.logger.Error("Could not read body", zap.Error(err))
		http.Error(w, "Failed to read request", http.StatusInternalServerError)
		return
	}

	req := api.TrainRequest{}

	// read the train request
	err = json.Unmarshal(body, &req)
	if err != nil {
		c.logger.Error("Failed to parse the train request",
			zap.Error(err),
			zap.String("payload", string(body)))
		http.Error(w, "Failed to decode the request", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)

	// Forward the request to the scheduler
	c.sendRequestToScheduler(req, TrainTask)
}

func (c Controller) handleInferenceRequest(w http.ResponseWriter, r *http.Request)  {

}

// Handles the request to
func (c *Controller) handleDatasetRequest(w http.ResponseWriter, r *http.Request)  {

}

// Handle Kubernetes heartbeats
func (c *Controller) handleHealth(w http.ResponseWriter, r *http.Request)  {
	w.WriteHeader(http.StatusOK)
}

// Returns the functions used to handle requests
func (c *Controller) getHandler()  http.Handler {
	r := mux.NewRouter()
	r.HandleFunc("/train", c.handleTrainRequest).Methods("POST")
	r.HandleFunc("/infer", c.handleInferenceRequest).Methods("POST")
	r.HandleFunc("/dataset", c.handleDatasetRequest).Methods("POST")
	r.HandleFunc("/health", c.handleHealth).Methods("GET")

	return r
}

// Starts the Controller API to handle requests
func (c *Controller) Serve(port int)  {
	c.logger.Info("Starting controller API", zap.Int("port", port))
	addr := fmt.Sprintf(":%v", port)

	// start the server
	err := http.ListenAndServe(addr, c.getHandler())
	c.logger.Fatal("Controller quit", zap.Error(err))
}