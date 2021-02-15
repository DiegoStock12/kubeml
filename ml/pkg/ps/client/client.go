package client

import (
	"bytes"
	"encoding/json"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/pkg/errors"
	"go.uber.org/zap"
	"net/http"
	"strings"
)

type (
	Client struct {
		logger     *zap.Logger
		psUrl      string
		httpClient *http.Client
	}
)

// MakeClient creates a client for the parameterServer
func MakeClient(logger *zap.Logger, psUrl string) *Client {
	return &Client{
		logger:     logger.Named("ps-client"),
		psUrl:      strings.TrimSuffix(psUrl, "/"),
		httpClient: &http.Client{},
	}

}

// UpdateTask sends the parameters to the PS for the
// next epoch of a particular training job
func (c *Client) UpdateTask(task *api.TrainTask) error {
	url := c.psUrl + "/update/" + task.Job.JobId

	c.logger.Debug("Updating task", zap.String("url", url))

	body, err := json.Marshal(task.Job.State)
	if err != nil {
		return errors.Wrap(err, "could not marshal update request")
	}

	_, err = c.httpClient.Post(url, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return errors.Wrap(err, "could not send update to Parameter Server")
	}

	return nil

}

// StartTask sends a new task to the parameter server
func (c *Client) StartTask(task *api.TrainTask) error {
	url := c.psUrl + "/start"

	// send request
	body, err := json.Marshal(task)
	if err != nil {
		return errors.Wrap(err, "could not marshal json")
	}

	_, err = c.httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return errors.Wrap(err, "could not start new task")
	}

	return nil
}

// UpdateMetrics sends a new metric set to the parameter server from the Jobs
// so they can be exposed to prometheus
func (c *Client) UpdateMetrics(jobId string, update *api.MetricUpdate) error {
	url := c.psUrl + "/metrics/" + jobId

	body, err := json.Marshal(update)
	if err != nil {
		return errors.Wrap(err, "could not marshal metrics object")
	}

	_, err = c.httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return errors.Wrap(err, "could not send metrics to the ps")
	}

	return nil
}

// JobFinished communicates to the parameter server that a job has finished. The PS
// will then clear its index, metrics and also communicate with the Scheduler
func (c *Client) JobFinished(jobId string, exitErr error) error {
	url := c.psUrl + "/finish/" + jobId

	var req *http.Request
	if exitErr != nil{
		body := []byte(exitErr.Error())
		req, _ = http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	} else {
		req, _ = http.NewRequest(http.MethodPost, url, nil)
	}

	_, err := c.httpClient.Do(req)
	if err != nil {
		return errors.Wrap(err, "error sending delete request")
	}

	return nil
}
