package client

import (
	"encoding/json"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/pkg/errors"
	"io/ioutil"
	"net/http"
)

func (c *Client) ListHistories() ([]api.History, error) {
	url := c.controllerUrl + "/history/list"

	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil , errors.Wrap(err, "could not perform history request")
	}
	defer resp.Body.Close()

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, "could not parse body")
	}

	var histories []api.History
	err = json.Unmarshal(data, &histories)
	if err != nil {
		return nil, errors.Wrap(err, "could not unmarshal json")
	}

	return histories, nil
}


// GetHistory returns the training history of a certain task
func (c *Client) GetHistory(taskId string) (string, error) {
	url := c.controllerUrl + "/history/get/" + taskId

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

func (c *Client) DeleteHistory(taskId string)  error  {
	url := c.controllerUrl + "/history/delete/" + taskId

	req, err := http.NewRequest(http.MethodDelete, url, nil)
	if err != nil {
		return errors.Wrap(err, "could not create request body")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return errors.Wrap(err, "could not handle request")
	}

	if resp.StatusCode != 200 {
		defer resp.Body.Close()
		res, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return err
		}
		return errors.New(string(res))
	}

	return nil
}

