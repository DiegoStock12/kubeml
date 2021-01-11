package client

import (
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"net/http"
)

type (
	Client struct {
		controllerUrl string
		httpClient *http.Client
	}
)

// MakeClient gets the kubernetes config and gets the IP address of the
// controller
// TODO for now just reference the local controller
func MakeClient() *Client {
	return &Client{
		controllerUrl: fmt.Sprintf("http://%s:%d", "localhost", api.CONTROLLER_DEBUG_PORT),
		httpClient:    &http.Client{},
	}
}


