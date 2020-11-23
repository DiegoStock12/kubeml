package scheduler

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/diegostock12/thesis/ml/pkg/model"
	"go.uber.org/zap"
	"net/http"
	"sync"
)

// Parameter server is run in a separate goroutine from the scheduler
// It can communicate with the scheduler through channels
type (
	ParameterServer struct {
		logger *zap.Logger

		// Id of the ps to give to the invoked functions
		psId        string
		parallelism int

		// tasks still to be completed
		toFinish int

		// Reference model that is trained
		model *model.Model

		// Channel to communicate with the scheduler and the API to receive layer names
		schedChan chan<- *ScheduleRequest
		layerChan <-chan []string
		epochChan <-chan *EpochFinished

		// Communication with the redisAI db
		redisClient *redisai.Client

		// Train request created for the PS
		req *api.TrainRequest

		// To get and set the value of the number of tasks to finish
		numLock sync.Mutex
	}

	EpochFinished struct{}
)

// Creates a new parameter server to train the model
func NewPS(logger *zap.Logger, id string, parallelism int,
	req *api.TrainRequest, schedChan chan<- *ScheduleRequest) *ParameterServer {

	logger.Info("Creating new parameter server")

	// TODO can this REDIS conn be shared with the model? I think so
	// Create the connection to the REDIS api that we'll pass through
	client := redisai.Connect(fmt.Sprintf("redis://%s:%d", api.RedisHost, api.RedisPort), nil)

	// Create the PS struct
	ps := &ParameterServer{
		logger:      logger.Named(fmt.Sprintf("ps-%s", id)),
		psId:        id,
		parallelism: parallelism,
		toFinish:    parallelism,
		schedChan:   schedChan,
		layerChan:   make(chan []string),
		epochChan:   make(chan *EpochFinished),
		redisClient: client,
		req:         req,
	}

	return ps

}

// Waits for the API to receive all the requests for starting the next epoch
// After this the ps needs to send a request to the scheduler to get the proper
// amount of functions to use in the next epoch
func (ps *ParameterServer) serveNextEpoch() {
	ps.logger.Info("Starting to wait for finish requests")

	// TODO here we should wait until the toFinish is 0 and then ask the scheduler for more or relaunch
	<-ps.epochChan

	ps.logger.Info("Epoch finished, contacting Scheduler")

	respChan := make(chan *ScheduleResponse)
	ps.schedChan <- &ScheduleRequest{
		psId:        ps.psId,
		network:     ps.req.ModelType,
		parallelism: ps.parallelism,
		respChan:    respChan,
	}

	ps.logger.Debug("Waiting for scheduler response")
	resp := <-respChan

	ps.logger.Info("Received next config from the Scheduler",
		zap.Int("new parallelism", resp.newParallelism))

	if resp.err != nil {
		ps.logger.Fatal("Error scheduling the new request", zap.Error(resp.err))
	}

	// Change the new limits
	ps.parallelism = resp.newParallelism
	ps.numLock.Lock()
	ps.toFinish = resp.newParallelism
	ps.numLock.Unlock()

	// Update the model and invoke the functions
	err := ps.model.Save()
	if err != nil {
		ps.logger.Error("Could not update model",
			zap.Error(err))
	}
	ps.invokeFunctions(ps.parallelism)

}



// TODO see how to handle correctly the fact that the response will not return
// Invokes N functions to start the next epoch
func (ps *ParameterServer) invokeFunctions(n int)  {
	for i := 0; i <n; i++ {
		go http.Get(fmt.Sprintf("%s/%s",api.ROUTER_ADDRESS, ps.req.FunctionName))
	}
}

// Starts a New parameter server which will execute the tasks
//1) start the new functions
//2) receive the notifications from the PS API about functions that have finished processing
//which will trigger the execution retrieval of gradients and the update of the model
//3) Start the API to get the requests from the functions

func (ps *ParameterServer) Start(port int) {
	ps.logger.Info("Started new parameter server", zap.Int("port", port))

	// Start the API to receive requests
	go ps.Serve(port)

	// Fetch the layers from the API
	ps.logger.Info("Waiting for the layer names")
	layers := <-ps.layerChan

	ps.logger.Debug("Received layers", zap.Any("Layers", layers))

	// TODO Should create model. Create a dummy model for now
	ps.logger.Debug("Creating random server that will go to the redis")
	m := model.NewModel(ps.logger, ps.psId, "resnet", layers, ps.req.LearningRate, ps.redisClient)
	// Set the model in the ps

	ps.model = m
	ps.logger.Debug("Building model")
	err := m.Build()
	if err != nil {
		panic(err)
	}

	// Summary of the model
	m.Summary()
	ps.logger.Info("Created parameter server")

	go ps.serveNextEpoch()

}
