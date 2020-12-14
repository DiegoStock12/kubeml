package controller

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/pkg/errors"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"
)

// scheduleRequest sends the request to the scheduler, the scheduler
// generates the id and returns it to the controller, which also returns it to the
// request submitter
func (c *Controller) scheduleRequest(req interface{}, task TaskType) (string, error){

	schedulerURL := fmt.Sprintf("%s:%d", api.DEBUG_URL, api.SCHEDULER_DEBUG_PORT)


	switch task {
	case TrainTask:
		// build the url
		url := schedulerURL + api.SCHEDULER_TRAIN_ENDPOINT
		c.logger.Debug("Sending train request to scheduler at", zap.String("url", url))
		// Create the request body
		reqBody, err := json.Marshal(req.(api.TrainRequest))
		if err != nil {
			c.logger.Error("Could not marshall the request",
				zap.Any("body", req),
				zap.Error(err))
		}
		// Send the request and return the id
		id, err := send(reqBody, url)
		return id, err

	case InferenceTask:
		url := schedulerURL + api.SCHEDULER_INFERENCE_ENDPOINT
		c.logger.Debug("Sending inference request to scheduler at", zap.String("url", url))
		// Create the request body
		reqBody, err := json.Marshal(req.(api.InferRequest))
		if err != nil {
			c.logger.Error("Could not marshall the request",
				zap.Any("body", req),
				zap.Error(err))
		}
		// Send the request and return the id
		id, err := send(reqBody, url)
		return id, err

	default:
		return "", errors.Errorf("Unknown request type", zap.Any("req", req))

	}
}


// send submits the request to the scheduler
// and returns the response as a string and an error if needed
func send(body []byte, url string) (string, error) {

	resp, err := http.Post(url, "application/json", bytes.NewBuffer(body))
	defer resp.Body.Close()

	if err != nil {
		return "", err
	}

	id, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	return string(id), nil

}
