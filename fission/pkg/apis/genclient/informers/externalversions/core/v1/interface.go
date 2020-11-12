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

// Code generated by informer-gen. DO NOT EDIT.

package v1

import (
	internalinterfaces "github.com/fission/fission/pkg/apis/genclient/informers/externalversions/internalinterfaces"
)

// Interface provides access to all the informers in this group version.
type Interface interface {
	// CanaryConfigs returns a CanaryConfigInformer.
	CanaryConfigs() CanaryConfigInformer
	// Environments returns a EnvironmentInformer.
	Environments() EnvironmentInformer
	// Functions returns a FunctionInformer.
	Functions() FunctionInformer
	// HTTPTriggers returns a HTTPTriggerInformer.
	HTTPTriggers() HTTPTriggerInformer
	// KubernetesWatchTriggers returns a KubernetesWatchTriggerInformer.
	KubernetesWatchTriggers() KubernetesWatchTriggerInformer
	// MessageQueueTriggers returns a MessageQueueTriggerInformer.
	MessageQueueTriggers() MessageQueueTriggerInformer
	// Packages returns a PackageInformer.
	Packages() PackageInformer
	// TimeTriggers returns a TimeTriggerInformer.
	TimeTriggers() TimeTriggerInformer
}

type version struct {
	factory          internalinterfaces.SharedInformerFactory
	namespace        string
	tweakListOptions internalinterfaces.TweakListOptionsFunc
}

// New returns a new Interface.
func New(f internalinterfaces.SharedInformerFactory, namespace string, tweakListOptions internalinterfaces.TweakListOptionsFunc) Interface {
	return &version{factory: f, namespace: namespace, tweakListOptions: tweakListOptions}
}

// CanaryConfigs returns a CanaryConfigInformer.
func (v *version) CanaryConfigs() CanaryConfigInformer {
	return &_canaryConfigInformer{factory: v.factory, namespace: v.namespace, tweakListOptions: v.tweakListOptions}
}

// Environments returns a EnvironmentInformer.
func (v *version) Environments() EnvironmentInformer {
	return &_environmentInformer{factory: v.factory, namespace: v.namespace, tweakListOptions: v.tweakListOptions}
}

// Functions returns a FunctionInformer.
func (v *version) Functions() FunctionInformer {
	return &_functionInformer{factory: v.factory, namespace: v.namespace, tweakListOptions: v.tweakListOptions}
}

// HTTPTriggers returns a HTTPTriggerInformer.
func (v *version) HTTPTriggers() HTTPTriggerInformer {
	return &_hTTPTriggerInformer{factory: v.factory, namespace: v.namespace, tweakListOptions: v.tweakListOptions}
}

// KubernetesWatchTriggers returns a KubernetesWatchTriggerInformer.
func (v *version) KubernetesWatchTriggers() KubernetesWatchTriggerInformer {
	return &_kubernetesWatchTriggerInformer{factory: v.factory, namespace: v.namespace, tweakListOptions: v.tweakListOptions}
}

// MessageQueueTriggers returns a MessageQueueTriggerInformer.
func (v *version) MessageQueueTriggers() MessageQueueTriggerInformer {
	return &_messageQueueTriggerInformer{factory: v.factory, namespace: v.namespace, tweakListOptions: v.tweakListOptions}
}

// Packages returns a PackageInformer.
func (v *version) Packages() PackageInformer {
	return &_packageInformer{factory: v.factory, namespace: v.namespace, tweakListOptions: v.tweakListOptions}
}

// TimeTriggers returns a TimeTriggerInformer.
func (v *version) TimeTriggers() TimeTriggerInformer {
	return &_timeTriggerInformer{factory: v.factory, namespace: v.namespace, tweakListOptions: v.tweakListOptions}
}
