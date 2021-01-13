package ps

import (
	"fmt"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
)

type (
	JobMetrics map[string]prometheus.Gauge
	JobHistory map[string][]float64
)

func NewHistory() JobHistory {
	return JobHistory{
		"loss":           {},
		"accuracy":       {},
		"trainLoss":      {},
		"parallelism":    {},
		"epoch_duration": {},
	}
}

// createAndRegisterMetrics creates and registers the metrics that the
// train job will publish during the training process for prometheus
// to scrape.
//
// The default metrics are trainLoss, loss, accuracy, parallelism and epoch duration
func (job *TrainJob) createAndRegisterMetrics() JobMetrics {

	jobId := job.jobId

	valLoss := promauto.NewGauge(prometheus.GaugeOpts{
		Name: fmt.Sprintf("kubeml_trainjob_%s_validation_loss", jobId),
		Help: fmt.Sprintf("Validation loss of train job %s", jobId),
	})

	accuracy := promauto.NewGauge(prometheus.GaugeOpts{
		Name: fmt.Sprintf("kubeml_trainjob_%s_validation_accuracy", jobId),
		Help: fmt.Sprintf("Validation accuracy of train job %s", jobId),
	})

	trainLoss := promauto.NewGauge(prometheus.GaugeOpts{
		Name: fmt.Sprintf("kubeml_trainjob_%s_train_loss", jobId),
		Help: fmt.Sprintf("Training loss of train job %s", jobId),
	})

	parallelism := promauto.NewGauge(prometheus.GaugeOpts{
		Name: fmt.Sprintf("kubeml_trainjob_%s_parallelism", jobId),
		Help: fmt.Sprintf("Parallelism level of train job %s", jobId),
	})

	epochDuration := promauto.NewGauge(prometheus.GaugeOpts{
		Name: fmt.Sprintf("kubeml_trainjob_%s_epoch_duration", jobId),
		Help: fmt.Sprintf("Epoch duration of train job %s", jobId),
	})

	return JobMetrics{
		"loss":           valLoss,
		"accuracy":       accuracy,
		"trainLoss":      trainLoss,
		"parallelism":    parallelism,
		"epoch_duration": epochDuration,
	}
}

// unregisterMetrics unregisters the metrics endpoints after the
// job is done training
func (job *TrainJob) unregisterMetrics() {
	for _, collector := range job.metrics {
		prometheus.Unregister(collector)
	}
}

// updateMetrics gets the history of the job and updates the
// exposed metrics accordingly. History and metrics should
// both have the same keys for a 1 to 1 correspondence
//
// Get the last value present in the history and update the corresponding
// prometheus gauge
func (job *TrainJob) updateMetrics() {
	for name, values := range job.history {
		if len(values) > 0 {
			job.metrics[name].Set(values[len(values)-1])
		} else {
			job.logger.Debug("Skipping metric with no results",
				zap.String("metric", name))
		}
	}
}
