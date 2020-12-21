package client

import (
	"bytes"
	"encoding/json"
	"github.com/diegostock12/thesis/ml/pkg/api"
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
	url := c.psUrl + "/update/" + task.JobId

	c.logger.Debug("Updating task", zap.String("url", url))

	body, err := json.Marshal(task)
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
func (c *Client) StartTask(task *api.TrainTask)  error {
	url := c.psUrl + "/start"

	// send request
	body, err := json.Marshal(task)
	if err != nil {
		return errors.Wrap(err,"could not marshal json")
	}

	_, err = c.httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil{
		return errors.Wrap(err, "could not start new task")
	}

	return nil
}


