package old

import (
	"fmt"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/diegostock12/kubeml/ml/pkg/util"
	"net/http"
)

// TODO change this to read the config file from kubernetes
const (
	controllerAddrKube = "192.168.99.101"
	controllerPortKube = 30457
)

type (
	Client struct {
		controllerUrl string
		httpClient *http.Client
	}
)

// MakeClient gets the kubernetes config and gets the IP address of the controller
func MakeClient() *Client {

	var controllerUrl string
	if util.IsDebugEnv() {
		controllerUrl = fmt.Sprintf("http://%s:%d", "localhost", api.CONTROLLER_DEBUG_PORT)
	} else {
		controllerUrl = fmt.Sprintf("http://%s:%d", controllerAddrKube, controllerPortKube)
	}

	fmt.Println("Using controller address", controllerUrl)

	return &Client{
		controllerUrl: controllerUrl,
		httpClient:    &http.Client{},
	}
}


