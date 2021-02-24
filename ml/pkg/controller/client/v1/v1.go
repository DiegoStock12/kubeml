package v1

import "net/http"

type V1Interface interface {
	NetworkGetter
	DatasetsGetter
	HistoryGetter
	TaskGetter
}

type V1 struct {
	controllerUrl string
	httpClient    *http.Client
}

func MakeV1Client(serverUrl string) V1Interface {
	return &V1{
		controllerUrl: serverUrl,
		httpClient:    &http.Client{},
	}
}

func (c *V1) Histories() HistoryInterface {
	return newHistories(c)
}

func (c *V1) Networks() NetworkInterface {
	return newNetworks(c)
}

func (c *V1) Datasets() DatasetInterface {
	return newDatasets(c)
}

func (c *V1) Tasks() TaskInterface {
	return newTasks(c)
}
