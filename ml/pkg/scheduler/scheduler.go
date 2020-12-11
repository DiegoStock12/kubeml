package scheduler

import (
	"github.com/diegostock12/thesis/ml/pkg/api"
	"go.uber.org/zap"
	"time"
)

type (
	Scheduler struct {
		logger *zap.Logger

		// TODO see how we'll handle multiple requests coming from multiple parameter servers
		// TODO how to decide how many functions each PS will invoke (we need metrics for this, start with constant)

		// queue that will hold the tasks
		queue SchedulerQueue

		// cache holds the time taken by the functions to complete
		// in their previous epoch. We will use this time to increase
		// or decrease the number of functions for a specific job
		// todo also allow to delete cache items
		cache map[string]float64
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
			s.logger.Warn("Schedule queue is empty, sleeping...")
			// If there is no element sleep
			// TODO see if the lock is a bottleneck
			time.Sleep(10 * time.Millisecond)
			continue
		}

		s.logger.Debug("Serving task", zap.Any("task", t))


		// based on the previous time seen, give
		// a new parallelism setting.
		// if first time give a -1, if there is a -1
		// it just ran 1 epoch so give same parallelism
		elapsed, exists := s.cache[t.JobId]
		if !exists {
			s.cache[t.JobId] = -1
			t.Parallelism = api.DEBUG_PARALLELISM
		} else {
			if elapsed == -1 {
				s.cache[t.JobId] = t.ElapsedTime
				// keep the parallelism untouched
			} else {
				// from epoch 3 we have the results already and we can scale
				// if the most recent time is not worse, increase parallelism
				// if the response time got worse decrease the parallelism
				if t.ElapsedTime <= elapsed {
					t.Parallelism++
				} else {
					t.Parallelism--
				}
			}
		}

		s.updateTask(t)

	}
}

// Start starts all of the goroutines that will take care of the proper
// functioning of the scheduler
// 1) Find next parallelism
// 2) Fetch metrics
// 3) **maybe look for the failed tasks queue? If the router fails
// keep the task there and retry
// 4) API so the scheduler is reachable from the other components
func Start(logger *zap.Logger, port int) {

	// Create the scheduler
	s := &Scheduler{
		logger: logger.Named("scheduler"),
		queue:  SchedulerQueue{},
		cache:  make(map[string]float64),
	}

	// Start consuming metrics and also listening for requests
	go s.consumeMetrics()
	go s.scheduleTasks()

	// Finally start the API
	go s.Serve(port)

	// Sleep for a couple of second
	s.logger.Debug("sleeping")
	time.Sleep(2 * time.Second)

	// Send a request into our own channel so we can create a PS
	s.logger.Debug("Sending random trainrequest")
	s.queue.pushRequest(&api.TrainRequest{
		ModelType:    "resnet",
		BatchSize:    128,
		Epochs:       3,
		Dataset:      "MNIST",
		LearningRate: 0.01,
		FunctionName: "network",
	})

}
