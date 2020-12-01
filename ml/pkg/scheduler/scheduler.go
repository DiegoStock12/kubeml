package scheduler

import (
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/diegostock12/thesis/ml/pkg/util"

	"go.uber.org/zap"
)

type (
	Scheduler struct {

		logger *zap.Logger

		// TODO see how we'll handle multiple requests coming from multiple parameter servers
		// TODO how to decide how many functions each PS will invoke (we need metrics for this, start with constant)
		// Schedule requests coming from the API
		apiChan chan *api.TrainRequest
		psChan chan *ScheduleRequest

		// TODO might need some kind of map to hold all the different running tasks and also metrics

	}

	ScheduleRequest struct {
		psId string
		network string

		parallelism int
		respChan chan *ScheduleResponse
	}

	ScheduleResponse struct {
		newParallelism int
		err error
	}
)

// Periodically consuming metrics to make scheduling decisions
func (s *Scheduler) consumeMetrics()  {
	// TODO unimplemented
	s.logger.Info("Scheduler starting to observe metrics")
	select { }
}

// Listen for the messages from the API and schedule the job
// The scheduler here has to
//1) Create parameter server for the task with a new id
//2) Indicate to the parameter server the right amount of functions to invoke
//(It is gonna be a constant of 3 for the start)
func (s *Scheduler) satisfyAPIRequests()  {
	s.logger.Info("Scheduler started satisfying the requests from the API")

	for {

		// receive requests from the channel coming from the API
		req := <-s.apiChan

		// Right now for testing just print that we got the request and start a parameter server
		s.logger.Info("Received request to schedule network", zap.String("model", req.ModelType))

		// generate a random uid for the psId and a random free port
		//psId := uuid.New().String()[:8]
		psId := "example"
		port, err := util.FindFreePort()
		if err != nil {
			s.logger.Error("Could not find free port",
				zap.Error(err))
		}

		//  TODO this parallelism should be optimized
		// Create a parameter server and start it in a new goroutine
		paramServer := NewPS(s.logger, psId, api.DEBUG_PARALLELISM, req, s.psChan)

		go paramServer.Start(port)

	}

}

// Loop satisfying the requests coming from the Parameter servers
// PS will send a request to the scheduler at the end of an epoch
// to get the number of functions that should be run in the next iteration
func (s *Scheduler) satisfyPSRequests() {
	s.logger.Info("Scheduler started satisfying the requests from the Parameter Servers")

	for {

		// Wait for requests from the PS
		req := <- s.psChan

		// TODO this should pack all the intelligence in terms of scheduling requests
		// For now just answer with a constant of 2 (same as parallelism)

		s.logger.Info("Received request from PS", zap.String("psID", req.psId))
		req.respChan <- &ScheduleResponse{
			newParallelism: 2,
			err:            nil,
		}
	}
}

// Start all the needed goroutines for
//1) periodically consume the metrics that will be used for taking decisions
//2) get the requests from the API through a channel and start the parameter server on demand
//3) Start the API so the functions can notify about status
func StartScheduler(logger *zap.Logger, port int) error {

	// Create the scheduler
	s := &Scheduler{
		logger:    logger.Named("scheduler"),
		apiChan: make(chan *api.TrainRequest),
		psChan: make(chan *ScheduleRequest),
	}

	// Start consuming metrics and also listening for requests
	go s.consumeMetrics()
	go s.satisfyAPIRequests()
	go s.satisfyPSRequests()


	// Finally start the API
	go s.Serve(port)


	// Send a request into our own channel so we can create a PS
	s.logger.Debug("Sending random trainrequest")
	s.apiChan <- &api.TrainRequest{
		ModelType:    "resnet",
		BatchSize:    128,
		Epochs:       1,
		Dataset:      "MNIST",
		LearningRate: 0.01,
		FunctionName: "network",
	}


	return nil

}


