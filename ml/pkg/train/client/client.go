package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/pkg/errors"
	"go.uber.org/zap"
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
		logger: logger.Named("trainJob-client"),
		httpClient: &http.Client{},
	}
}

// UpdateTask sends the updated parameters to the TrainJob
func (c *Client) UpdateTask(task *api.TrainTask)  error{
	jobIP := task.Job.Pod.Status.PodIP
	url := fmt.Sprintf("http://%v:%v/update", jobIP, jobApiPort)

	// send just the job state to the job
	body, err := json.Marshal(task.Job.State)
	if err != nil {return errors.Wrap(err, "could not marshal state")}

	_, err = c.httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {return errors.Wrap(err, "could not send update to job")}

	return nil
}

// UpdateTask sends the updated parameters to the TrainJob
func (c *Client) StartTask(task *api.TrainTask)  error{
	jobIP := task.Job.Pod.Status.PodIP
	url := fmt.Sprintf("http://%v:%v/start", jobIP, jobApiPort)

	// send just the job state to the job
	body, err := json.Marshal(task)
	if err != nil {return errors.Wrap(err, "could not marshal task")}

	_, err = c.httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {return errors.Wrap(err, "could not send task to job")}

	return nil
}



