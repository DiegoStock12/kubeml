package ps

import (
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

type (
	TaskType string
)

const (
	TrainTask     TaskType = "train"
	InferenceTask TaskType = "infer"
)

var (
	metricsAddr = ":8080"

	// labelsJob are used those exported by each trainJob, to
	// monitor its current state during the training phase
	labelsJob = []string{"jobid"}

	// labelsPS are exported by the parameter server to monitor the number
	// of tasks of each type (training, inference) currently happening
	labelsPS = []string{"type"}

	// Metrics for the job
	// validation and train loss,
	// accuracy,
	// parallelism and duration of each epoch
	valLoss = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "kubeml_job_validation_loss",
			Help: "Validation loss of a train job",
		},
		labelsJob,
	)

	accuracy = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "kubeml_job_validation_accuracy",
			Help: "Validation accuracy of a train job",
		},
		labelsJob,
	)

	trainLoss = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "kubeml_job_train_loss",
			Help: "Train loss of a train job",
		},
		labelsJob,
	)

	parallelism = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "kubeml_job_parallelism",
			Help: "Parallelism of a train job",
		},
		labelsJob,
	)

	epochDuration = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "kubeml_job_epoch_duration_seconds",
			Help: "Epoch duration of a train job",
		},
		labelsJob,
	)

	// Parameter server level metrics
	tasksRunning = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "kubeml_job_running_total",
			Help: "Number of running tasks of each type",
		},
		labelsPS,
	)
)

func init() {
	tasksRunning.WithLabelValues("train").Set(0)
	tasksRunning.WithLabelValues("inference").Set(0)
}


// updateMetrics takes the history of the job and refreshes the
// ps metrics for that job using the jobId as the filtering label
func updateMetrics(jobId string, metrics api.MetricUpdate) {
	valLoss.WithLabelValues(jobId).Set(metrics.ValidationLoss)
	accuracy.WithLabelValues(jobId).Set(metrics.Accuracy)
	trainLoss.WithLabelValues(jobId).Set(metrics.TrainLoss)
	epochDuration.WithLabelValues(jobId).Set(metrics.EpochDuration)
	parallelism.WithLabelValues(jobId).Set(metrics.Parallelism)
}

// clearMetrics deletes the metrics associated with a jobId after
// the training process is done
func clearMetrics(jobId string) {
	valLoss.DeleteLabelValues(jobId)
	accuracy.DeleteLabelValues(jobId)
	trainLoss.DeleteLabelValues(jobId)
	parallelism.DeleteLabelValues(jobId)
	epochDuration.DeleteLabelValues(jobId)
}

// taskStarted updates the gauges for tasks in currently
// running in the parameter server
func taskStarted(t TaskType) {

	switch t {
	case TrainTask:
		tasksRunning.WithLabelValues("train").Inc()
	case InferenceTask:
		tasksRunning.WithLabelValues("inference").Inc()

	}

}

// taskFinished updates the gauges for tasks in currently
// running in the parameter server when a task is concluded
func taskFinished(t TaskType) {

	switch t {
	case TrainTask:
		tasksRunning.WithLabelValues("train").Dec()
	case InferenceTask:
		tasksRunning.WithLabelValues("inference").Dec()
	}

}
