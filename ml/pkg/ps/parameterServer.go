package ps

import (
	"errors"
	"github.com/diegostock12/thesis/ml/pkg/api"
	schedulerClient "github.com/diegostock12/thesis/ml/pkg/scheduler/client"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"net/http"
	"sync"
	"time"
)

const (
	KubeMlNamespace = "kubeml"
	KubeMlContainer = "diegostock12/kubeml"
)


// Parameter server is run in a separate goroutine from the scheduler
// It can communicate with the scheduler through channels
type (

	// ParameterServer is the main parameter server that spins out the worker
	// jobs that serve training jobs.
	// The parameter server saves the index to communicate with each of the jobs.
	// The jobs send requests to the Scheduler when they finish an epoch, and the
	// scheduler responds to the main PS api, which then makes the response reach
	// the correct train job.
	ParameterServer struct {
		logger *zap.Logger

		port int

		// scheduler is the client used by the PS and all
		// its train jobs to send requests to the scheduler
		scheduler *schedulerClient.Client

		// kubernetes client to handle created pods for jobs
		kubeClient *kubernetes.Clientset

		// jobIndex with all the train jobs
		// when receiving a response from the scheduler the
		// api will consult the index and send the response to
		// the appropriate worker
		jobIndex map[string]*api.TrainTask

		// doneChan simply receives the exit messages from the jobs to
		// delete them from the index.
		// The jobs send their ID to the channel and the ps deletes the history
		doneChan chan string

		// Lock to alter the index
		// TODO should it be RW?
		mu sync.Mutex
	}
)


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

// createJobPod creates a pod for a new train job with a specific ID
func (ps *ParameterServer) createJobPod(task *api.TrainTask) error {

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "job-" + task.Job.JobId,
			Namespace: KubeMlNamespace,
			Labels: map[string]string{
				"svc": "job",
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:            "job",
					Image:           KubeMlContainer,
					ImagePullPolicy: corev1.PullIfNotPresent,
					Command:         []string{"/kubeml"},
					Args:            []string{"--jobPort", "9090"},
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
								Port:   intstr.IntOrString{Type: intstr.Int, IntVal: 9090, StrVal: "9090"},
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
		ps.logger.Error("Error creating pod for training job",
			zap.Error(err))
		return err
	}

	err = ps.waitForPodRunning(podRef, 20 * time.Second)
	if err != nil {
		ps.logger.Error("Error waiting for pod to start",
			zap.Error(err))
		return err
	}

	ps.logger.Debug("Created pod")

	// send the train task to the pod



	return nil

}

func serveMetrics(logger *zap.Logger) {

	logger.Debug("Serving metrics")
	// Expose the prometheus metrics endpoint
	http.Handle("/metrics", promhttp.Handler())
	err := http.ListenAndServe(metricsAddr, nil)

	logger.Fatal("metrics endpoint exited", zap.Error(err))

}

// receiveFinish  waits in a channel made available to all jobs at creation time
// to receive signals that they are finished.
//
// After receiving this signal, clear the job entry in the job index, send a message
// to the scheduler so that the cache is also updated and decrement the number of
// running tasks in the metric
func (ps *ParameterServer) receiveFinish() {

	for {
		id := <-ps.doneChan

		ps.mu.Lock()
		if _, exists := ps.jobIndex[id]; exists {
			ps.logger.Debug("Received finish message", zap.String("jobId", id))
			delete(ps.jobIndex, id)

			// delete it as well from the scheduler cache
			err := ps.scheduler.FinishJob(id)
			if err != nil {
				ps.logger.Error("Error deleting the job from the scheduler cache",
					zap.Error(err),
					zap.String("jobId", id))
			}

		} else {
			ps.logger.Warn("Received finish message from unknown job",
				zap.String("jobId", id))
		}

		ps.mu.Unlock()
		ps.taskFinished(TrainTask)

	}

}

// Start Starts a New parameter server which will execute the tasks
//1) start the new functions
//2) receive the notifications from the PS API about functions that have finished processing
//which will trigger the execution retrieval of gradients and the update of the model
//3) Start the API to get the requests from the functions
func Start(logger *zap.Logger, port int, schedulerUrl string) {

	// build the PS
	ps := &ParameterServer{
		logger:   logger.Named("ps"),
		port:     port,
		jobIndex: make(map[string]*api.TrainTask),
		doneChan: make(chan string),
	}

	// set the scheduler client
	ps.scheduler = schedulerClient.MakeClient(ps.logger, schedulerUrl)

	ps.logger.Info("Started new parameter server")

	// start the listener for job finishes
	go ps.receiveFinish()
	go serveMetrics(ps.logger)

	// Start the API to receive requests
	ps.Serve(port)
}
