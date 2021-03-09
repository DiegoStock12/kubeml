package v1

import (
	"encoding/json"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/pkg/errors"
	"io/ioutil"
	"net/http"
)

type (
	HistoryGetter interface {
		Histories() HistoryInterface
	}

	HistoryInterface interface {
		Get(taskId string) (*api.History, error)
		Delete(taskId string) error
		List() ([]api.History, error)
	}

	histories struct {
		controllerUrl string
		httpClient    *http.Client
	}
)

func newHistories(c *V1) HistoryInterface {
	return &histories{
		controllerUrl: c.controllerUrl,
		httpClient:    c.httpClient,
	}
}

func (h *histories) Get(taskId string) (*api.History, error) {
	url := h.controllerUrl + "/history/" + taskId

	resp, err := h.httpClient.Get(url)
	if err != nil {
		return nil, errors.Wrap(err, "could not perform history request")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// read the http error string and return it
		errString, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return nil, errors.Wrap(err, "an error occurred")
		}
		return nil, errors.New(string(errString))
	}

	// return the json parsed to string
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, "could not parse body")
	}

	var history api.History
	err = json.Unmarshal(body, &history)
	if err != nil {
		return nil, errors.Wrap(err, "could not unmarshal history")
	}

	return &history, nil
}

func (h *histories) Delete(taskId string) error {
	url := h.controllerUrl + "/history/" + taskId

	req, err := http.NewRequest(http.MethodDelete, url, nil)
	if err != nil {
		return errors.Wrap(err, "could not create request body")
	}

	resp, err := h.httpClient.Do(req)
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

func (h *histories) List() ([]api.History, error) {
	url := h.controllerUrl + "/history"

	resp, err := h.httpClient.Get(url)
	if err != nil {
		return nil, errors.Wrap(err, "could not perform history request")
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
