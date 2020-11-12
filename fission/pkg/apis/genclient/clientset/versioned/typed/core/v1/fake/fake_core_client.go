/*
Copyright The Fission Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Code generated by client-gen. DO NOT EDIT.

package fake

import (
	v1 "github.com/fission/fission/pkg/apis/genclient/clientset/versioned/typed/core/v1"
	rest "k8s.io/client-go/rest"
	testing "k8s.io/client-go/testing"
)

type FakeCoreV1 struct {
	*testing.Fake
}

func (c *FakeCoreV1) CanaryConfigs(namespace string) v1.CanaryConfigInterface {
	return &FakeCanaryConfigs{c, namespace}
}

func (c *FakeCoreV1) Environments(namespace string) v1.EnvironmentInterface {
	return &FakeEnvironments{c, namespace}
}

func (c *FakeCoreV1) Functions(namespace string) v1.FunctionInterface {
	return &FakeFunctions{c, namespace}
}

func (c *FakeCoreV1) HTTPTriggers(namespace string) v1.HTTPTriggerInterface {
	return &FakeHTTPTriggers{c, namespace}
}

func (c *FakeCoreV1) KubernetesWatchTriggers(namespace string) v1.KubernetesWatchTriggerInterface {
	return &FakeKubernetesWatchTriggers{c, namespace}
}

func (c *FakeCoreV1) MessageQueueTriggers(namespace string) v1.MessageQueueTriggerInterface {
	return &FakeMessageQueueTriggers{c, namespace}
}

func (c *FakeCoreV1) Packages(namespace string) v1.PackageInterface {
	return &FakePackages{c, namespace}
}

func (c *FakeCoreV1) TimeTriggers(namespace string) v1.TimeTriggerInterface {
	return &FakeTimeTriggers{c, namespace}
}

// RESTClient returns a RESTClient that is used to communicate
// with API server by this client implementation.
func (c *FakeCoreV1) RESTClient() rest.Interface {
	var ret *rest.RESTClient
	return ret
}
