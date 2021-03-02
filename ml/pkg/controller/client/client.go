package client

import (
	"fmt"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	v1 "github.com/diegostock12/kubeml/ml/pkg/controller/client/v1"
	"github.com/diegostock12/kubeml/ml/pkg/util"
)

// TODO change this to read the config file from kubernetes
const (
	controllerAddrKube = "192.168.99.101"
	controllerPortKube = 31156

	controllerAddrCloud = "34.69.78.66"
	controllerPortCloud = 80
)

type (
	Interface interface {
		V1() v1.V1Interface
		ServerUrl() string
	}

	KubemlClient struct {
		controllerUrl string
		v1            v1.V1Interface
	}
)

func MakeKubemlClient() (*KubemlClient, error) {

	var controllerUrl string
	var err error
	if util.IsDebugEnv() {
		controllerUrl = fmt.Sprintf("http://%s:%d", "localhost", api.ControllerPortDebug)
	} else {
		controllerUrl, err  = getControllerUrl()
		if err != nil {
			return nil, err
		}
	}

	fmt.Println("Using controller address", controllerUrl)

	return &KubemlClient{
		controllerUrl: controllerUrl,
		v1:            v1.MakeV1Client(controllerUrl),
	}, nil

}

func (c *KubemlClient) V1() v1.V1Interface {
	return c.v1
}

func (c *KubemlClient) ServerUrl() string {
	return c.controllerUrl
}
