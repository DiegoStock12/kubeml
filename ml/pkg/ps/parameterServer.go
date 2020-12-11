package ps

import (
	"github.com/diegostock12/thesis/ml/pkg/api"
	"go.uber.org/zap"
	"sync"
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

		// jobIndex with all the train jobs
		// when receiving a response from the scheduler the
		// api will consult the index and send the response to
		// the appropriate worker
		jobIndex map[string]chan *api.TrainTask

		mu *sync.Mutex
	}


)


// Start Starts a New parameter server which will execute the tasks
//1) start the new functions
//2) receive the notifications from the PS API about functions that have finished processing
//which will trigger the execution retrieval of gradients and the update of the model
//3) Start the API to get the requests from the functions
func  Start(logger *zap.Logger, port int) {

	// build the PS
	ps := &ParameterServer{
		logger:    logger.Named("ps"),
		port:      port,
		jobIndex:  make(map[string]chan *api.TrainTask),
		mu: &sync.Mutex{},
	}

	ps.logger.Info("Started new parameter server")


	// Start the API to receive requests
	ps.Serve(port)
}


