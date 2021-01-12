package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/gorilla/mux"
	"go.mongodb.org/mongo-driver/bson"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"
)

// Handle a train request and forward it to the scheduler
func (c *Controller) handleTrainRequest(w http.ResponseWriter, r *http.Request) {
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


	// Forward the request to the scheduler
	id, err := c.scheduler.SubmitTrainTask(req)
	if err != nil {
		c.logger.Error("Could not get job id",
			zap.Error(err))
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	c.logger.Debug("got job id",zap.String("id", id))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(id))
}

// handleInferenceRequest gets an Inference request from the client
// and simply sends the query to the scheduler
func (c *Controller) handleInferenceRequest(w http.ResponseWriter, r *http.Request) {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		c.logger.Error("Could not read inference request",
			zap.Error(err))
		http.Error(w, "Failed to read request", http.StatusInternalServerError)
		return
	}

	var req api.InferRequest
	// read the train request
	err = json.Unmarshal(body, &req)
	if err != nil {
		c.logger.Error("Failed to parse the train request",
			zap.Error(err),
			zap.String("payload", string(body)))
		http.Error(w, "Failed to decode the request", http.StatusInternalServerError)
		return
	}

	// Forward the request to the scheduler
	// we get the response object with the predictions as
	// a byte array already
	resp, err := c.scheduler.SubmitInferenceTask(req)
	if err != nil {
		c.logger.Error("Could not get job id",
			zap.Error(err))
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	c.logger.Debug("got response",zap.String("predictions", string(resp)))
	w.WriteHeader(http.StatusOK)
	w.Header().Set("Content-Type", "application/json")
	w.Write(resp)
}

// historyRequest gets a history from mongoDB
func (c *Controller) historyRequest(w http.ResponseWriter, r *http.Request)  {
	vars := mux.Vars(r)
	taskId := vars["taskId"]

	c.logger.Debug("Getting history", zap.String("taskId", taskId))

	// Use the mongo client to get the history
	var history api.History
	collection := c.mongoClient.Database("kubeml").Collection("history")
	err := collection.FindOne(context.TODO(), bson.M{"_id":taskId}).Decode(&history)
	if err != nil {
		c.logger.Error("Could not find history",
			zap.Error(err))
		http.Error(w, "Could not find history for request" ,http.StatusNotFound)
		return
	}

	resp, err := json.MarshalIndent(history, "", "  ")
	if err != nil {
		c.logger.Error("Could not marshal history",
			zap.Error(err))
		http.Error(w, "Error marshaling request" ,http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(resp)
}


// Handle Kubernetes heartbeats
func (c *Controller) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

// Returns the functions used to handle requests
func (c *Controller) getHandler() http.Handler {
	r := mux.NewRouter()
	r.HandleFunc("/train", c.handleTrainRequest).Methods("POST")
	r.HandleFunc("/infer", c.handleInferenceRequest).Methods("POST")
	r.HandleFunc("/health", c.handleHealth).Methods("GET")
	r.HandleFunc("/dataset/{name}", c.StorageServiceProxy)
	r.HandleFunc("/history/{taskId}", c.historyRequest).Methods("GET")

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
