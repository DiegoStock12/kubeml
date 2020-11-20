package scheduler

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"go.uber.org/zap"
	"sync"

	"github.com/diegostock12/thesis/ml/pkg/model"
)


// Parameter server is run in a separate goroutine from the scheduler
// It can communicate with the scheduler through channels
type (
	ParameterServer struct {

		logger *zap.Logger

		// Id of the ps to give to the invoked functions
		psId string
		parallelism int

		// tasks still to be completed
		toFinish int

		// Reference model that is trained
		model *model.Model

		// Channel to communicate with the scheduler
		schedChan chan<- *ScheduleRequest

		// Communication with the redisAI db
		redisClient redisai.Client

		// So only one thread can edit the model at a time
		modelLock sync.Mutex
	}
)

// Creates a new parameter server to train the model
func NewPS(logger *zap.Logger, id string, parallelism int,
	req *api.TrainRequest, schedChan chan<- *ScheduleRequest) *ParameterServer {

	logger.Info("Creating new parameter server")


	client := redisai.Connect(fmt.Sprintf("redis://%s:%d", api.RedisHost, api.RedisPort), nil)
	layerNames := []string{"conv1", "conv2", "fc1", "fc2"}

	// TODO Should create model. Create a dummy model for now

	logger.Debug("Creating random server that will go to the redis")
	m := model.NewModel(logger, "resnet", layerNames, 0.01, client)

	logger.Debug("Building model")
	err := m.Build("example")
	if err != nil {
		panic(err)
	}


	// Summary of the model
	m.Summary()



	ps := &ParameterServer{
		logger:      logger.Named(fmt.Sprintf("ps-%s", id)),
		psId:        id,
		parallelism: parallelism,
		toFinish: parallelism,
		model:       m,
		schedChan:   schedChan,
		redisClient: redisai.Client{},
	}


	ps.logger.Info("Created parameter server")

	return ps
}


// Starts a New parameter server which will execute the tasks
//1) start the new functions
//2) receive the notifications from the PS API about functions that have finished processing
//which will trigger the execution retrieval of gradients and the update of the model
//3) Start the API to get the requests from the functions
// TODO fill in this function
func (ps *ParameterServer) Start(port int)  {
	ps.logger.Info("Started new parameter server", zap.Int("port", port))


	go ps.Serve(port)

}
