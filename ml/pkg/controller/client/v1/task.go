package v1

import (
	"encoding/json"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"io/ioutil"
	"net/http"
)

type (
	TaskGetter interface {
		Tasks() TaskInterface
	}

	TaskInterface interface {
		List() ([]api.TrainTask, error)
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
