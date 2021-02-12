package client

import (
	"bytes"
	"encoding/json"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/pkg/errors"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"
	"strings"
)

type (

	// Client gives access
	Client struct {
		logger       *zap.Logger
		schedulerUrl string
		httpClient   *http.Client
	}
)

// MakeClient creates a client for the scheduler
func MakeClient(logger *zap.Logger, schedulerUrl string) *Client {
	return &Client{
		logger:       logger.Named("scheduler-client"),
		schedulerUrl: strings.TrimSuffix(schedulerUrl, "/"),
		httpClient:   &http.Client{},
	}
}

// UpdateJob sends a request to the scheduler to determine the new level
// of parallelism that should be given to a job based on metrics and
// previous epochs
func (c *Client) UpdateJob(task *api.TrainTask) error {
	url := c.schedulerUrl + "/job"

	body, err := json.Marshal(task)
	if err != nil {
		return errors.Wrap(err, "could not marshal request to update job")
	}

	_, err = c.httpClient.Post(url, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return errors.Wrap(err, "could not send request to scheduler")
	}

	return nil

}

// FinishJob makes the scheduler delete the job entry from the cache
func (c *Client) FinishJob(jobId string) error {
	url := c.schedulerUrl + "/finish/" + jobId

	req, err := http.NewRequest(http.MethodDelete, url, nil)
	if err != nil {
		return errors.Wrap(err, "could not send finish job request")
	}

	_, err = c.httpClient.Do(req)
	if err != nil {
		return errors.Wrap(err, "error performing finish request")
	}
	return nil
}

// SubmitTrainTask submits a training task to the scheduler
func (c *Client) SubmitTrainTask(req api.TrainRequest) (string, error) {
	url := c.schedulerUrl + "/train"

	c.logger.Debug("Sending train request to scheduler at", zap.String("url", url))
	// Create the request body
	reqBody, err := json.Marshal(req)
	if err != nil {
		return "", errors.Wrap(err, "could not send train request to scheduler")
	}
	// Send the request and return the id
	id, err := c.sendTask(reqBody, url)
	return id, err
}

// SubmitInferenceTask submits an inference task to the scheduler
// and returns the response from the inference task as a byte array
func (c *Client) SubmitInferenceTask(req []byte) ([]byte, error) {
	url := c.schedulerUrl + "/infer"

	// Send the request and return the id
	resp, err := c.httpClient.Post(url, "application/json", bytes.NewBuffer(req))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, "could not read response body")
	}

	return body, nil
}


// sendTask submits the request to the scheduler
// and returns the response as a string and an error if needed
func (c *Client) sendTask(body []byte, url string) (string, error) {

	resp, err := c.httpClient.Post(url, "application/json", bytes.NewBuffer(body))
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
