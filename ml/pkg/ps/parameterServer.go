package ps

import (
	"github.com/diegostock12/thesis/ml/pkg/api"
	schedulerClient "github.com/diegostock12/thesis/ml/pkg/scheduler/client"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
	"net/http"
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

		// scheduler is the client used by the PS and all
		// its train jobs to send requests to the scheduler
		scheduler *schedulerClient.Client

		// jobIndex with all the train jobs
		// when receiving a response from the scheduler the
		// api will consult the index and send the response to
		// the appropriate worker
		jobIndex map[string]chan *api.TrainTask

		// doneChan simply receives the exit messages from the jobs to
		// delete them from the index.
		// The jobs send their ID to the channel and the ps deletes the history
		doneChan chan string

		// Lock to alter the index
		// TODO should it be RW?
		mu sync.Mutex
	}
)

func serveMetrics(logger *zap.Logger) {

	logger.Debug("Serving metrics")
	// Expose the prometheus metrics endpoint
	http.Handle("/metrics", promhttp.Handler())
	err := http.ListenAndServe(metricsAddr, nil)

	logger.Fatal("metrics endpoint exited", zap.Error(err))

}

// receiveFinish simply waits in the
func (ps *ParameterServer) receiveFinish() {

	// Loop forever and wait for messages receiving the finish from the
	// jobs
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
		jobIndex: make(map[string]chan *api.TrainTask),
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
