package test

import (
	"fmt"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	v1 "github.com/diegostock12/kubeml/ml/pkg/controller/client/test/v1"
	"github.com/diegostock12/kubeml/ml/pkg/util"
)

// TODO change this to read the config file from kubernetes
const (
	controllerAddrKube = "192.168.99.101"
	controllerPortKube = 30457
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

func MakeKubemlClient() *KubemlClient {
	var controllerUrl string
	if util.IsDebugEnv() {
		controllerUrl = fmt.Sprintf("http://%s:%d", "localhost", api.CONTROLLER_DEBUG_PORT)
	} else {
		controllerUrl = fmt.Sprintf("http://%s:%d", controllerAddrKube, controllerPortKube)
	}

	fmt.Println("Using controller address", controllerUrl)

	return &KubemlClient{
		controllerUrl: controllerUrl,
		v1:            v1.MakeV1Client(controllerUrl),
	}

}

func (c *KubemlClient) V1() v1.V1Interface {
	return c.v1
}

func (c *KubemlClient) ServerUrl() string {
	return c.controllerUrl
}
