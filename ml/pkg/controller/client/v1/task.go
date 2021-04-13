package v1

import (
	"encoding/json"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/pkg/errors"
	"io/ioutil"
	"net/http"
)

type (
	TaskGetter interface {
		Tasks() TaskInterface
	}

	TaskInterface interface {
		List() ([]api.TrainTask, error)
		Stop(id string) error
	}

	tasks struct {
		controllerUrl string
		httpClient    *http.Client
	}
)

func newTasks(c *V1) TaskInterface {
	return &tasks{
		controllerUrl: c.controllerUrl,
		httpClient:    c.httpClient,
	}
}

func (t *tasks) List() ([]api.TrainTask, error) {
	url := t.controllerUrl + "/tasks"

	resp, err := t.httpClient.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var tasks []api.TrainTask
	err = json.Unmarshal(body, &tasks)
	if err != nil {
		return nil, err
	}

	return tasks, nil

}

func (t *tasks) Stop(id string) error {
	url := t.controllerUrl + "/tasks/" + id

	req, err := http.NewRequest(http.MethodDelete, url, nil)
	if err != nil {
		return errors.Wrap(err, "could not create request body")
	}

	resp, err := t.httpClient.Do(req)
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
