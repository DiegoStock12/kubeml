package ps

import (
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/hashicorp/go-multierror"
	"github.com/pkg/errors"
	"go.uber.org/zap"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"time"
)

const kubemlVersion = "0.1.2-avg"

func (ps *ParameterServer) isPodReady(podName string) wait.ConditionFunc {
	return func() (done bool, err error) {

		pod, err := ps.kubeClient.CoreV1().Pods(KubeMlNamespace).Get(podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		switch pod.Status.Phase {
		case corev1.PodRunning:
			return true, nil
		case corev1.PodFailed, corev1.PodSucceeded:
			return false, errors.New("pod failed or was succeeded")
		}

		return false, nil
	}
}

func (ps *ParameterServer) waitForPodRunning(pod *corev1.Pod, timeout time.Duration) error {
	return wait.PollImmediate(time.Second, timeout, ps.isPodReady(pod.Name))
}

// createJobResources creates the pod and service offered by a job
func (ps *ParameterServer) createJobResources(task api.TrainTask) (*corev1.Pod, *corev1.Service, error) {

	// create the pod
	pod, err := ps.createJobPod(task)
	if err != nil {
		return nil, nil, errors.Wrap(err, "error creating pod for job")
	}

	// create the service
	svc, err := ps.createJobService(task)
	if err != nil {
		ps.logger.Error("error creating service, deleting pod...")
		var e *multierror.Error
		e = multierror.Append(e, errors.Wrap(err, "error creating service"))

		err = ps.deleteJobPod(pod)
		e = multierror.Append(e, err)

		return nil, nil, e.ErrorOrNil()
	}

	return pod, svc, nil

}

// createJobService creates a service exposing the pod so it can be accessed by the functions
func (ps *ParameterServer) createJobService(task api.TrainTask) (*corev1.Service, error) {

	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "job-" + task.Job.JobId,
			Namespace: KubeMlNamespace,
			Labels: map[string]string{
				"svc": "job",
				"job": task.Job.JobId,
			},
		},
		Spec: corev1.ServiceSpec{
			// select as backing pod the job with the same
			// id in the labels
			Selector: map[string]string{
				"job": task.Job.JobId,
			},
			Type: corev1.ServiceTypeClusterIP,
			Ports: []corev1.ServicePort{
				{
					Port:       80,
					TargetPort: intstr.FromInt(9090),
				},
			},
		},
	}

	// create the service
	svcRef, err := ps.kubeClient.CoreV1().Services(KubeMlNamespace).Create(svc)
	if err != nil {
		return nil, err
	}

	return svcRef, nil

}

// createJobPod creates a pod for a new train job with a specific ID
func (ps *ParameterServer) createJobPod(task api.TrainTask) (*corev1.Pod, error) {

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "job-" + task.Job.JobId,
			Namespace: KubeMlNamespace,
			Labels: map[string]string{
				"svc": "job",
				"job": task.Job.JobId,
			},
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{
				{
					Name:            "job",
					Image:           KubeMlContainer,
					ImagePullPolicy: corev1.PullIfNotPresent,
					Command:         []string{"/kubeml"},
					Args: []string{
						"--jobPort",
						"9090",
						"--jobId",
						task.Job.JobId,
					},
					Ports: []corev1.ContainerPort{
						{
							Name:          "http",
							ContainerPort: 9090,
							Protocol:      "TCP",
						},
					},
					ReadinessProbe: &corev1.Probe{
						Handler: corev1.Handler{
							Exec: nil,
							HTTPGet: &corev1.HTTPGetAction{
								Path:   "/health",
								Port:   intstr.FromInt(9090),
								Scheme: "HTTP",
							},
						},
						InitialDelaySeconds: 1,
						TimeoutSeconds:      1,
						PeriodSeconds:       1,
						SuccessThreshold:    1,
						FailureThreshold:    30,
					},
				},
			},
		},
	}

	podRef, err := ps.kubeClient.CoreV1().Pods(KubeMlNamespace).Create(pod)
	if err != nil {
		return nil, err
	}

	ps.logger.Debug("data from pod",
		zap.Any("name", podRef.Name),
		zap.Any("ip", podRef.Status.PodIP),
		zap.Any("phase", podRef.Status.Phase))

	err = ps.waitForPodRunning(podRef, 20*time.Second)
	if err != nil {
		return nil, err
	}

	ps.logger.Debug("Created pod")

	// get the reference of the pod with the IP for creation of the client
	podRef, err = ps.kubeClient.CoreV1().Pods(KubeMlNamespace).Get(pod.Name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	return podRef, nil

}

// deleteJobResources deletes the pod and service used for a job
func (ps *ParameterServer) deleteJobResources(task *api.TrainTask) error {
	var err *multierror.Error

	e := ps.deleteJobService(task.Job.Svc)
	err = multierror.Append(err, e)

	e = ps.deleteJobPod(task.Job.Pod)
	err = multierror.Append(err, e)

	return err.ErrorOrNil()
}

// deleteJobService deletes the service for a train job
func (ps *ParameterServer) deleteJobService(svc *corev1.Service) error {
	err := ps.kubeClient.CoreV1().Services(KubeMlNamespace).Delete(svc.Name, &metav1.DeleteOptions{})
	return err
}

// deleteJobPod deletes the pod used for a job
func (ps *ParameterServer) deleteJobPod(pod *corev1.Pod) error {
	err := ps.kubeClient.CoreV1().Pods(KubeMlNamespace).Delete(pod.Name, &metav1.DeleteOptions{})
	return err
}
