package ps

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

type (

	// JobHistory saves the intermediate results from the training process
	// epoch to epoch
	JobHistory struct {
		ValidationLoss []float64
		Accuracy       []float64
		TrainLoss      []float64
		Parallelism    []float64
		EpochDuration  []float64
	}
)

// ToMap converts the job history to a more general
// map for serizlization
func (h JobHistory) ToMap() map[string][]float64 {
	return map[string][]float64{
		"validation_loss": h.ValidationLoss,
		"accuracy":        h.Accuracy,
		"train_loss":      h.TrainLoss,
		"parallelism":     h.Parallelism,
		"epoch_duration":  h.EpochDuration,
	}
}

var (

	metricsAddr = ":8080"

	// labelsJob are used those exported by each trainJob, to
	// monitor its current state during the training phase
	labelsJob = []string{"jobid"}

	// labelsPS are exported by the parameter server to monitor the number
	// of tasks of each type (training, validation) currently happening
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
	functionsRunning = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "kubeml_functions_running",
			Help: "Number of runnning functions",
		},
		labelsPS,
	)
)

func init() {
	functionsRunning.WithLabelValues("train").Set(0)
	functionsRunning.WithLabelValues("inference").Set(0)
}



func last(arr []float64) float64 {
	return arr[len(arr)-1]
}

// updateMetrics takes the history of the job and refreshes the
// ps metrics for that job using the jobId as the filtering label
func (job TrainJob) updateMetrics() {

	// The validation values might not be set yet since the validation
	// function runs independently from the training jobs
	if len(job.history.ValidationLoss) > 0 {
		valLoss.WithLabelValues(job.jobId).Set(last(job.history.ValidationLoss))
	}

	if len(job.history.Accuracy) > 0 {
		accuracy.WithLabelValues(job.jobId).Set(last(job.history.Accuracy))
	}

	// set the training values
	trainLoss.WithLabelValues(job.jobId).Set(last(job.history.TrainLoss))
	epochDuration.WithLabelValues(job.jobId).Set(last(job.history.EpochDuration))
	parallelism.WithLabelValues(job.jobId).Set(last(job.history.Parallelism))


}

// clearMetrics deletes the metrics associated with a jobId after
// the training process is done
func (job *TrainJob) clearMetrics() {
	valLoss.DeleteLabelValues(job.jobId)
	accuracy.DeleteLabelValues(job.jobId)
	trainLoss.DeleteLabelValues(job.jobId)
	parallelism.DeleteLabelValues(job.jobId)
	epochDuration.DeleteLabelValues(job.jobId)
}

// parametersUpdated is called when a new configuration for a task,
// be it that the task is created or the parallelism is updated
func (ps *ParameterServer) parametersUpdated()  {

}

