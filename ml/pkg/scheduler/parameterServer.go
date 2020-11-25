package scheduler

import (
	"encoding/json"
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/diegostock12/thesis/ml/pkg/model"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"sync"
)

// Parameter server is run in a separate goroutine from the scheduler
// It can communicate with the scheduler through channels
type (
	ParameterServer struct {
		logger *zap.Logger

		// Id of the ps to give to the invoked functions
		psId        string
		psPort      int
		parallelism int

		// tasks still to be completed
		toFinish int
		epoch    int

		// Reference model that is trained
		model *model.Model

		// Channel to communicate with the scheduler and the API to receive layer names
		schedChan chan<- *ScheduleRequest
		layerChan chan []string
		epochChan chan struct{}

		// Communication with the redisAI db
		redisClient *redisai.Client

		// Train request created for the PS
		req *api.TrainRequest

		// To get and set the value of the number of tasks to finish
		numLock sync.Mutex

		// map to save the time series of results (similar to keras history)
		// maps the label (accuracy, loss, val_loss...) to an array indexed by the epochs
		// TODO doesn't need lock cause only one val func will be running at once?
		history map[string][]float32
	}
)

// NewPS Creates a new parameter server to train the model
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
		epoch:       1,
		schedChan:   schedChan,
		layerChan:   make(chan []string),
		epochChan:   make(chan struct{}),
		redisClient: client,
		req:         req,
		history: make(map[string][]float32),
	}

	return ps

}

// serveTrainJob is the main Waits for the API to receive all the requests for starting the next epoch
// After this the ps needs to send a request to the scheduler to get the proper
// amount of functions to use in the next epoch
func (ps *ParameterServer) serveTrainJob() {
	ps.logger.Info("Starting to serve train job")

	// Loop for as many epochs as required by the request
	for ps.epoch <= ps.req.Epochs {

		ps.logger.Info("Started new epoch",
			zap.Int("epoch", ps.epoch))

		// Invoke the functions
		// TODO we could do the thing of adding an extra b funcs to deal with stragglers
		ps.invokeTrainFunctions(ps.parallelism)

		// The model updates and so on is handled in parallel in the API
		// Here we just wait for all functions to be done
		// TODO here we should wait until the toFinish is 0 and then ask the scheduler for more or relaunch
		<-ps.epochChan

		ps.logger.Info("Epoch finished, saving model")
		// update the model and invoke the functions
		err := ps.model.Save()
		if err != nil {
			ps.logger.Error("Could not update model",
				zap.Error(err))
		}

		// TODO handle the response from the val func
		// Invoke the validation function while we wait for the scheduler
		go ps.invokeValFunction()

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

	}

	ps.logger.Info(fmt.Sprintf("Training finished after %d epochs", ps.epoch))

	// TODO should save results of the training in the database

}

// invokeInitFunction calls a single function which initializes the
// model, saves it to the database and returns the layer names that the ps will save
func (ps *ParameterServer) invokeInitFunction() ([]string, error) {
	query := ps.buildFunctionURL(0, 1, "init", ps.req.FunctionName)
	resp, err := http.Get(query)

	if err != nil {
		// TODO here we should implement retries like in the fetcher specialize in fission
		// TODO maybe create a special function called execute with retries
		ps.logger.Error("Could not call the init function",
			zap.String("funcName", ps.req.FunctionName),
			zap.Any("request", ps.req),
			zap.Error(err))

		return nil, err
	}

	var names []string
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		ps.logger.Fatal("Could not read layer names",
			zap.Error(err))

		return nil, err
	}

	_ = json.Unmarshal(data, &names)

	// Set the layer names
	return names, nil

}

// invokeTrainFunctions Invokes N functions to start the next epoch
// TODO see how to handle correctly the fact that the response will not return
func (ps *ParameterServer) invokeTrainFunctions(n int) {
	for i := 0; i < n; i++ {
		query := ps.buildFunctionURL(i, n, "train", ps.req.FunctionName)

		// TODO this should return the train accuracy and loss for all of them
		go http.Get(query)
	}
}

// invokeValFunction After getting all the gradients and publishing the new model invoke
// the validation function to get the performance of the system, these are returned as a dict
func (ps *ParameterServer) invokeValFunction() {

	// TODO instead of returning the map we could add it to a PS level map that tracks the progress
	var results map[string]float32

	query := ps.buildFunctionURL(0, 1, "val", ps.req.FunctionName)
	resp, err := http.Get(query)
	if err != nil {
		// TODO here we should implement retries like in the fetcher specialize in fission
		// TODO maybe create a special function called execute with retries
		ps.logger.Error("Could not call the init function",
			zap.String("funcName", ps.req.FunctionName),
			zap.Any("request", ps.req),
			zap.Error(err))
	}

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		ps.logger.Error("Could not read layer names",
			zap.Error(err))

	}

	// Unmarshall the JSON to a dict
	// This JSON should give accuracy, precision, recall...
	_ = json.Unmarshal(data, &results)

	// Update the history with the new results
	for metric := range ps.history {
		value, exists := ps.history[metric]
		if exists {
			ps.history[metric] = append(value, results[metric])
		} else {
			ps.history[metric] = []float32{results[metric]}
		}
	}

}

// Start Starts a New parameter server which will execute the tasks
//1) start the new functions
//2) receive the notifications from the PS API about functions that have finished processing
//which will trigger the execution retrieval of gradients and the update of the model
//3) Start the API to get the requests from the functions
func (ps *ParameterServer) Start(port int) {
	ps.logger.Info("Started new parameter server", zap.Int("port", port))

	// set the psPort in the object
	ps.psPort = port

	// Start the API to receive requests
	go ps.Serve(port)

	// Fetch the layers from the API
	ps.logger.Info("Waiting for the layer names")

	// Here we could do
	//layers := <-ps.layerChan
	layers, err := ps.invokeInitFunction()
	if err != nil {
		panic(err)
	}

	ps.logger.Debug("Received layers", zap.Any("Layers", layers))

	// TODO Should create model. Create a dummy model for now
	ps.logger.Debug("Creating random server that will go to the redis")
	m := model.NewModel(ps.logger, ps.psId, "resnet", layers, ps.req.LearningRate, ps.redisClient)
	// Set the model in the ps

	ps.model = m
	ps.logger.Debug("Building model")
	err = m.Build()
	if err != nil {
		panic(err)
	}

	// Summary of the model
	m.Summary()
	ps.logger.Info("Created parameter server")

	go ps.serveTrainJob()

}

// TODO this should take something to determine the batch of the data that should be used
// buildFunctionURL returns the url that the PS will invoke to execute the function
func (ps *ParameterServer) buildFunctionURL(funcId, numFunc int, task, funcName string) string {

	values := url.Values{}
	values.Set("task", task)
	values.Set("psId", ps.psId)
	values.Set("psPort", strconv.Itoa(ps.psPort))
	values.Set("N", strconv.Itoa(numFunc))
	values.Set("funcId", strconv.Itoa(funcId))

	dest := api.ROUTER_ADDRESS + "/" + funcName + "?" + values.Encode()

	ps.logger.Debug("Built url", zap.String("url", dest))

	return dest
}
