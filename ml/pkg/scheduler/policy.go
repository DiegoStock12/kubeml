package scheduler

import (
	"github.com/diegostock12/thesis/ml/pkg/api"
	"go.uber.org/zap"
	"sync"
)

const (
	ThroughPutScaleDownThreshold = 1.2
	ThroughputScaleUpThreshold   = 1.05
)

// SchedulerPolicy defines the methods needed to be implemented by the scheduler
// in order to support the tasks from KubeML, these involve how to calculate the parallelism
// of the task based on previous performance of the task
type (
	SchedulerPolicy interface {
		// calculate paralellism returns the parallelism for the next epoch
		calculateParallelism(task api.TrainTask) (parallelism int, op TaskOperation)
		taskFinished(taskId string)
	}

	ThroughputBasedPolicy struct {
		logger *zap.Logger

		// timeCache saves the throughput from previous epochs
		// of the different jobs and is used to reactively scale up or down
		// the parallelism when we see that the time elapsed for an epoch
		// increases or decreases
		timeCache map[string]float64

		mu *sync.RWMutex
	}
)

func makeThroughputPolicy(logger *zap.Logger) ThroughputBasedPolicy {
	return ThroughputBasedPolicy{
		logger:    logger.Named("throughput-policy"),
		timeCache: make(map[string]float64),
		mu:        &sync.RWMutex{},
	}
}

// calculateParallelism for the throughput based policy simply scales up if the performance
// is better or slightly worse than in previous epochs (given by the scale-up threshold), and scales
// down if the performance is much worse.
//
// In between those thresholds the parallelism is kept untouched
func (tp ThroughputBasedPolicy) calculateParallelism(task api.TrainTask) (parallelism int, op TaskOperation) {

	tp.mu.RLock()
	prevTime, exists := tp.timeCache[task.Job.JobId]
	tp.mu.RUnlock()

	// If it is the first epoch and we do not have a history
	// of this task, simply return the debug parallelism
	if !exists {
		tp.mu.Lock()
		tp.timeCache[task.Job.JobId] = 0
		tp.mu.Unlock()

		return api.DEBUG_PARALLELISM, CreateTask

	} else {

		switch {
		case prevTime == 0:
			tp.logger.Debug("No previous time, increasing parallelism")
			tp.timeCache[task.Job.JobId] = task.Job.State.ElapsedTime
			return task.Job.State.Parallelism + 1, UpdateTask

		// If the new time is better than the prevTime
		// always scale up and set a new reference time
		case task.Job.State.ElapsedTime <= prevTime*ThroughputScaleUpThreshold:
			tp.logger.Debug("Time is better, scaling up")
			tp.timeCache[task.Job.JobId] = task.Job.State.ElapsedTime
			return task.Job.State.Parallelism + 1, UpdateTask

		// If the performance is much worse (20%) than the reference
		// time, downscale and set a new reference time
		case task.Job.State.ElapsedTime >= prevTime*ThroughPutScaleDownThreshold:
			tp.logger.Debug("Time is worse, scaling down")
			tp.timeCache[task.Job.JobId] = task.Job.State.ElapsedTime
			return task.Job.State.Parallelism - 1, UpdateTask

		default:
			tp.logger.Debug("Time is worse within the limits, keeping parallelism")
			return task.Job.State.Parallelism, UpdateTask
		}

	}

}

// taskFinished handles the finish of the task, here simply deletes it from
// the time cache
func (tp ThroughputBasedPolicy) taskFinished(taskId string) {
	tp.mu.Lock()
	defer tp.mu.Unlock()
	delete(tp.timeCache, taskId)
}
