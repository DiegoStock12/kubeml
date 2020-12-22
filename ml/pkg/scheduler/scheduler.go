package scheduler

import (
	"github.com/diegostock12/thesis/ml/pkg/api"
	psClient "github.com/diegostock12/thesis/ml/pkg/ps/client"
	"go.uber.org/zap"
	"sync"
	"time"
)

const (
	scaleDownLimit = 1.2
	scaleUpLimit   = 1.05
)

type (
	Scheduler struct {
		logger *zap.Logger

		// queue that will hold the tasks
		queue SchedulerQueue

		// ps is the client to send requests
		// and updates to the parameter server
		ps *psClient.Client

		// cache holds the time taken by the functions to complete
		// in their previous epoch. We will use this time to increase
		// or decrease the number of functions for a specific job
		// todo also allow to delete cache items
		cache map[string]float64

		// lock to read and write the map
		mu sync.RWMutex
	}
)

// Periodically consuming metrics to make scheduling decisions
func (s *Scheduler) consumeMetrics() {
	// TODO unimplemented
	s.logger.Info("Scheduler starting to observe metrics")
	select {}
}

// scheduleTasks Loop satisfying the requests coming from the Parameter servers
// PS will send a request to the scheduler at the end of an epoch
// to get the number of functions that should be run in the next iteration
func (s *Scheduler) scheduleTasks() {
	s.logger.Info("Scheduler started satisfying the requests from the Parameter Servers")

	for {

		// Wait until there is an element in the queue
		t, err := s.queue.popTask()
		if err != nil {
			//s.logger.Warn("Schedule queue is empty, sleeping...")
			// If there is no element sleep
			// TODO see if the lock is a bottleneck
			time.Sleep(50 * time.Millisecond)
			continue
		}

		s.logger.Debug("Serving task", zap.Any("task", t))

		// based on the previous time seen, give
		// a new parallelism setting.
		// if first time give a -1, if there is a -1
		// it just ran 1 epoch so give same parallelism
		s.mu.RLock()
		threshold, exists := s.cache[t.JobId]
		s.mu.RUnlock()

		if !exists {
			s.mu.Lock()
			s.cache[t.JobId] = -1
			s.mu.Unlock()

			t.Parallelism = api.DEBUG_PARALLELISM

			err = s.ps.StartTask(t)
			if err != nil {
				s.logger.Error("Error starting task",
					zap.Any("task", t),
					zap.Error(err))
			}

		} else {
			s.calculateParallelism(t, threshold)

			err = s.ps.UpdateTask(t)
			if err != nil {
				s.logger.Error("Error updating task",
					zap.Any("task", t))
			}
		}

	}
}

// calculateParallelism receives a task and retu
func (s *Scheduler) calculateParallelism(task *api.TrainTask, threshold float64) {
	// If it is the first epoch we still do not have a previous time
	// so just set it and keep it untouched
	s.mu.Lock()
	defer s.mu.Unlock()
	// by default scale up even if the time is a bit worse than before
	// (allow 5% worse results to still scale up)
	// if the result is more than 20% worse, downscale for the next epoch

	switch {
	// increase the parallelism by 1 to compare with the previous
	// if it is worse, downscale
	case threshold == -1:
		s.logger.Debug("No previous time, increasing parallelism")
		task.Parallelism++
		s.cache[task.JobId] = task.ElapsedTime

	// If the new time is better than the threshold
	// always scale up and set a new reference time
	case task.ElapsedTime <= threshold*scaleUpLimit:
		s.logger.Debug("Time is better, scaling up")
		task.Parallelism++
		s.cache[task.JobId] = task.ElapsedTime

	// If the performance is much worse (20%) than the reference
	// time, downscale and set a new reference time
	case task.ElapsedTime >= threshold*scaleDownLimit:
		s.logger.Debug("Time is worse, scaling down")
		task.Parallelism--
		s.cache[task.JobId] = task.ElapsedTime

	// In other cases keep the same parallelism and reference time,
	// even though the time might be worse
	default:
		s.logger.Debug("Time is worse within the limits, keeping parallelism")
	}

}

// Start starts all of the goroutines that will take care of the proper
// functioning of the scheduler
// 1) Find next parallelism
// 2) Fetch metrics
// 3) **maybe look for the failed tasks queue? If the router fails
// keep the task there and retry
// 4) API so the scheduler is reachable from the other components
func Start(logger *zap.Logger, port int, psUrl string) {

	// Create the scheduler
	s := &Scheduler{
		logger: logger.Named("scheduler"),
		queue:  NewQueue(),
		cache:  make(map[string]float64),
	}

	// set the ps client
	s.ps = psClient.MakeClient(s.logger, psUrl)

	// Start consuming metrics and also listening for requests
	go s.consumeMetrics()
	go s.scheduleTasks()

	// Finally start the API
	s.Serve(port)

}
