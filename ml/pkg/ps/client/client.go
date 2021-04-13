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

// StopTask stops the task given the task id
func (c *Client) StopTask(id string) error {
	url := c.psUrl + "/stop/" + id

	req, err := http.NewRequest(http.MethodDelete, url, nil)
	if err != nil {
		return errors.Wrap(err, "could not create request body")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return errors.Wrap(err, "could not handle request")
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		res, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return err
		}
		return errors.New(string(res))
	}

	return nil

}

// ListTasks returns the response of the tasks in a byte format
// since the usage will only be internally, the controller will just redirect the bytes
// to the requester
func (c *Client) ListTasks() ([]byte, error) {
	url := c.psUrl + "/tasks"

	c.logger.Debug("Listing tasks")

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, errors.Wrap(err, "error performing request")
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, "error reading response body")
	}

	return body, nil
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

	var err error
	// if there is an error add it in the body so that the
	// parameter server reports it
	if exitErr != nil {
		body := []byte(exitErr.Error())
		_, err = c.httpClient.Post(url, "text/plain", bytes.NewReader(body))
	} else {
		_, err = c.httpClient.Post(url, "text/plain", nil)
	}

	if err != nil {
		return errors.Wrap(err, "could not send finish notification")
	}

	return nil
}
