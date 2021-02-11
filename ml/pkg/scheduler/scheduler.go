package scheduler

import (
	psClient "github.com/diegostock12/thesis/ml/pkg/ps/client"
	"go.uber.org/zap"
	"time"
)

const (
	scaleDownLimit = 1.2
	scaleUpLimit   = 1.05
)

type (
	TaskOperation int

	Scheduler struct {
		logger *zap.Logger

		// queue that will hold the tasks
		queue SchedulerQueue

		// ps is the client to send requests
		// and updates to the parameter server
		ps *psClient.Client

		// SchedulerPolicy to determine the task parallelism
		policy SchedulerPolicy
	}
)

// Declare the task operation
const (
	CreateTask TaskOperation = iota
	UpdateTask
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
		task, err := s.queue.popTask()
		if err != nil {
			//s.logger.Warn("Schedule queue is empty, sleeping...")
			// If there is no element sleep
			// TODO see if the lock is a bottleneck
			time.Sleep(10 * time.Millisecond)
			continue
		}

		s.logger.Debug("Serving task", zap.Any("task", task))

		// calculate the parallelism of the next epoch using the scheduler policy
		parallelism, operation := s.policy.calculateParallelism(*task)

		// TODO if the scheduling fails, retry as K8s does by putting it in the queue
		task.Job.State.Parallelism = parallelism
		switch operation {
		case CreateTask:
			err = s.ps.StartTask(task)
			if err != nil {
				s.logger.Error("Error sending task creation request to parameter server",
					zap.Any("task", task),
					zap.Error(err))
			}

		case UpdateTask:
			err = s.ps.UpdateTask(task)
			if err != nil {
				s.logger.Error("Error sending task update request to parameter server",
					zap.Any("task", task),
					zap.Error(err))
			}
		}

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
	}

	// set the ps client
	s.ps = psClient.MakeClient(s.logger, psUrl)
	s.policy = makeThroughputPolicy(s.logger)

	// Start consuming metrics and also listening for requests
	go s.consumeMetrics()
	go s.scheduleTasks()

	// Finally start the API
	s.Serve(port)

}
