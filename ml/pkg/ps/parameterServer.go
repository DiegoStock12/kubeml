package ps

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/diegostock12/thesis/ml/pkg/scheduler"
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
		schedChan chan<- *scheduler.ScheduleRequest

		// Communication with the redisAI db
		redisClient redisai.Client

		// So only one thread can edit the model at a time
		modelLock sync.Mutex
	}
)

// Creates a new parameter server to train the model
func NewPS(logger *zap.Logger, id string, parallelism int,
	req *api.TrainRequest, schedChan chan<- *scheduler.ScheduleRequest) *ParameterServer  {

	logger.Info("Creating new parameter server")

	// TODO Should create model. Create a dummy model for now
	m := &model.Model{
		Name:       req.ModelType,
		LayerNames: nil,
		Layers:     nil,
		Lr:         req.LearningRate,
	}



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

}
