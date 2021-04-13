package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/pkg/errors"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"
)

const jobApiPort = 9090

type (
	Client struct {
		logger     *zap.Logger
		httpClient *http.Client
	}
)

func MakeClient(logger *zap.Logger) *Client {
	return &Client{
		logger:     logger.Named("trainJob-client"),
		httpClient: &http.Client{},
	}
}

// Stop stops the running task
func (c *Client) Stop(task *api.TrainTask) error {
	svcName := task.Job.Svc.Name
	url := fmt.Sprintf("http://%v/stop", svcName)

	req, err := http.NewRequest(http.MethodDelete, url, nil)
	if err != nil {
		return errors.Wrap(err, "could not create request body")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return errors.Wrap(err, "could not stop task")
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

// UpdateTask sends the updated parameters to the TrainJob
func (c *Client) UpdateTask(task *api.TrainTask, update api.JobState) error {
	svcName := task.Job.Svc.Name
	url := fmt.Sprintf("http://%v/update", svcName)

	// send just the job state to the job
	body, err := json.Marshal(update)
	if err != nil {
		return errors.Wrap(err, "could not marshal state")
	}

	_, err = c.httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return errors.Wrap(err, "could not send update to job")
	}

	return nil
}

// UpdateTask sends the updated parameters to the TrainJob
func (c *Client) StartTask(task *api.TrainTask) error {
	svcName := task.Job.Svc.Name
	url := fmt.Sprintf("http://%v/start", svcName)
	c.logger.Debug("starting task", zap.String("url", url))

	// send just the job state to the job
	body, err := json.Marshal(task)
	if err != nil {
		return errors.Wrap(err, "could not marshal task")
	}

	resp, err := c.httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return errors.Wrap(err, "could not send task to job")
	}

	if resp.StatusCode != http.StatusOK {
		c.logger.Warn("Start task returned bad code")
		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			c.logger.Error("error reading body", zap.Error(err))
			return nil
		}

		err = errors.New(string(body))
		c.logger.Error("returned body:", zap.Error(err))
		return err
	}

	return nil
}
