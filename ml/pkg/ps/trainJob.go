package ps

import (
	"errors"
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/diegostock12/thesis/ml/pkg/model"
	"go.uber.org/zap"
	"sync"
	"time"
)

type (
	// TrainJob is each of the workers launched by the parameter server.
	// The worker is responsible from managing the reference model, saving the
	// intermediate accuracy/validation results in the history, and requesting/receiving
	// new scheduling responses from the scheduler
	TrainJob struct {
		logger *zap.Logger

		jobId       string
		parallelism int

		// parameters of the trainjob
		toFinish int32
		epoch    int

		// reference model
		model *model.Model

		// schedChan receives messages from the PS
		schedChan <-chan *api.TrainTask
		doneChan chan<- string
		// epochChan is to synchronize when receiving all the responses from
		// the functions
		//// TODO we can just wait until all the functions are ready with a WG
		//epochChan chan struct{}

		// Communicate with the cache
		redisClient *redisai.Client

		// request which has to be satisfied
		task *api.TrainTask

		// history of the train job
		history map[string][]float32

		// to avoid exiting without the validation tasks finish
		wgVal *sync.WaitGroup
	}
)

// newTrainJob Creates a new TrainJob that will take care of a specific train request
func newTrainJob(logger *zap.Logger,
	task *api.TrainTask, schedChan <-chan *api.TrainTask, doneChan chan string) *TrainJob {

	logger.Info("Creating new train job")

	// Create the connection to the REDIS api that we'll pass through to the PS
	client := redisai.Connect(fmt.Sprintf("redis://%s:%d", api.REDIS_ADDRESS_DEBUG, api.REDIS_PORT_DEBUG), nil)

	// Create the PS struct
	job := &TrainJob{
		logger:      logger.Named(fmt.Sprintf("trainJob-%s", task.JobId)),
		jobId:       task.JobId,
		parallelism: task.Parallelism,
		epoch:       1,
		schedChan:   schedChan,
		doneChan: doneChan,
		redisClient: client,
		task:        task,
		history:     make(map[string][]float32),
		wgVal:       &sync.WaitGroup{},
	}

	return job

}

// initializeModel launches the function and creates the model used by the TrainJob
func (job *TrainJob) initializeModel() error {
	job.logger.Debug("Calling init function")
	layers, err := job.invokeInitFunction()
	if err != nil {
		return err
	}
	if len(layers) == 0 {
		return errors.New("length of the layers is zero")
	}

	job.logger.Debug("Received layers", zap.Any("layers", layers))
	job.logger.Debug("Creating model")
	m := model.NewModel(job.logger, job.jobId, job.task.Parameters, layers, job.redisClient)
	job.model = m

	job.logger.Debug("Building model...")
	err = m.Build()
	if err != nil {
		return err
	}

	// Summary of the model
	m.Summary()
	job.logger.Info("Initialized train job")

	return nil
}

// serveTrainJob is the main Waits for the API to receive all the requests for starting the next epoch
// After this the job needs to send a request to the scheduler to get the proper
// amount of functions to use in the next epoch
func (job *TrainJob) serveTrainJob() {

	job.logger.Info("Starting to serve train job")
	job.logger.Info("Initializing model")

	err := job.initializeModel()
	if err != nil {
		job.logger.Fatal("Could not initialize model",
			zap.Error(err))
	}

	// Loop for as many epochs as required by the request
	for ;job.epoch <= job.task.Parameters.Epochs; job.epoch++ {

		job.logger.Info("Started new epoch",
			zap.Int("epoch", job.epoch))
		startTime := time.Now()

		job.invokeTrainFunctions()

		// Get the elapsed time
		elapsed := time.Now().Sub(startTime)


		job.logger.Debug("Elapsed time",
			zap.Any("elapsed", elapsed))
		job.logger.Info("Epoch finished, saving model")


		// update the model and invoke the functions
		err := job.model.Save()
		if err != nil {
			job.logger.Fatal("Could not update model",
				zap.Error(err))
		}

		// Invoke the validation function while we wait for the scheduler
		job.wgVal.Add(1)
		go job.invokeValFunction(job.wgVal)

		// TODO send request to the scheduler through the API
		job.sendSchedulerRequest(elapsed)

		job.logger.Debug("Waiting for scheduler response")
		resp := <-job.schedChan

		job.logger.Info("Received next config from the Scheduler",
			zap.Int("new parallelism", resp.Parallelism))

		if resp.Parallelism < 1 {
			job.logger.Error("Received bad configuration from the scheduler",
				zap.Int("parallelism", resp.Parallelism))
		}

		job.task = resp

		// Change the new limits
		job.parallelism = resp.Parallelism
	}

	// Wait for the val functions to finish
	job.wgVal.Wait()

	job.logger.Info(fmt.Sprintf("Training finished after %d epochs", job.epoch-1))

	// TODO should save results of the training in the database
	//job.saveTrainingHistory()
	job.logger.Info("Exiting...", zap.Any("history", job.history))

	// Send the id to the PS so it can delete the
	// entry in the job index
	job.doneChan <- job.jobId

}




