package controller

import (
	"encoding/json"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"
)

// Handle a train request and forward it to the scheduler
func (c *Controller) train(w http.ResponseWriter, r *http.Request) {
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

// infer gets an Inference request from the client
// and simply sends the query to the scheduler
func (c *Controller) infer(w http.ResponseWriter, r *http.Request) {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		c.logger.Error("Could not read inference request",
			zap.Error(err))
		http.Error(w, "Failed to read request", http.StatusInternalServerError)
		return
	}

	// Instead of unmarshalling and marshalling again the
	// request, send the body as is to improve performance
	resp, err := c.scheduler.SubmitInferenceTask(body)
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
