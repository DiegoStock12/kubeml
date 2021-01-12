package client

import (
	"github.com/pkg/errors"
	"io/ioutil"
)

// getHistory returns the training history of a certain task
func (c *Client) GetHistory(taskId string) (string, error) {
	url := c.controllerUrl + "/history/" + taskId

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return "" , errors.Wrap(err, "could not perform history request")
	}
	defer resp.Body.Close()

	// return the json parsed to string
	history, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", errors.Wrap(err, "could not parse body")
	}

	return string(history), nil
}
