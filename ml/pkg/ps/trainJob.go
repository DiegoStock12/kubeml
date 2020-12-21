package ps

import (
	"errors"
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/diegostock12/thesis/ml/pkg/model"
	schedulerClient "github.com/diegostock12/thesis/ml/pkg/scheduler/client"
	"github.com/hashicorp/go-multierror"
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

		// client for the scheduler (shared by all trainjobs)
		scheduler *schedulerClient.Client

		jobId       string
		parallelism int

		// parameters of the trainjob
		epoch int

		// reference model
		model *model.Model

		// optimizer for the model
		optimizer model.ParallelSGD

		// schedChan receives messages from the PS
		schedChan <-chan *api.TrainTask
		doneChan  chan<- string

		// Communicate with the cache
		redisClient *redisai.Client

		// request which has to be satisfied
		task *api.TrainTask

		// history of the train job
		history map[string][]float32

		// to avoid exiting without the validation tasks finish
		wgVal *sync.WaitGroup
	}

	// funcResults holds the function id and the execution
	// results of a function, be it a training or validation function
	funcResults struct {
		funcId  int
		results map[string]float32
	}
)

// newTrainJob Creates a new TrainJob that will take care of a specific train request
func newTrainJob(logger *zap.Logger,
	task *api.TrainTask,
	schedChan <-chan *api.TrainTask,
	doneChan chan string,
	client *schedulerClient.Client) *TrainJob {

	logger.Info("Creating new train job")

	// Create the connection to the REDIS api that we'll pass through to the PS
	redisClient := redisai.Connect(fmt.Sprintf(
		"redis://%s:%d", api.REDIS_ADDRESS_DEBUG, api.REDIS_PORT_DEBUG), nil)

	// Create the PS struct
	// TODO allow for more optimizers than the SGD
	job := &TrainJob{
		logger:      logger.Named(fmt.Sprintf("trainJob-%s", task.JobId)),
		scheduler:   client,
		jobId:       task.JobId,
		parallelism: task.Parallelism,
		epoch:       1,
		schedChan:   schedChan,
		doneChan:    doneChan,
		redisClient: redisClient,
		task:        task,
		history:     make(map[string][]float32),
		wgVal:       &sync.WaitGroup{},
	}

	job.optimizer = model.MakeParallelSGD(job.logger)

	return job

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
	for ; job.epoch <= job.task.Parameters.Epochs; job.epoch++ {

		// call all the training functions,
		// gather the stats and return the time taken in the
		// iteration
		elapsed, err := job.train()
		if err != nil {
			job.logger.Error("Error training model", zap.Error(err))
		}

		// Invoke the validation function
		job.validate()

		// If it is not our last epoch send the request to
		// the scheduler so we can get the new parameters
		if job.epoch < job.task.Parameters.Epochs {

			job.task.ElapsedTime = elapsed.Seconds()
			err = job.scheduler.UpdateJob(job.task)
			if err != nil {
				job.logger.Error("Error updating parallelism",
					zap.Error(err))
				continue
			}

			job.logger.Debug("Waiting for scheduler response")
			resp := <-job.schedChan

			job.logger.Info("Received next config from the Scheduler",
				zap.Int("new parallelism", resp.Parallelism))
			if resp.Parallelism < 1 {
				job.logger.Error("Received bad configuration from the scheduler",
					zap.Int("parallelism", resp.Parallelism))
			}

			job.task = resp
			job.parallelism = resp.Parallelism
		}
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

// updateModel optimizes the model's weights and biases with the gradients
// saved by the functions in the previous epoch
//func (job *TrainJob) updateModel(funcs ...int) error {
//	job.logger.Info("Updating model",
//		zap.Any("funcs", funcs))
//
//	var result *multierror.Error
//	N := len(funcs)
//
//	// For each of the functions call the optimizer step
//	// function to fetch the gradients and update the model weights
//	for _, id := range funcs {
//		err := job.optimizer.Step(job.model, id, N)
//		result = multierror.Append(result, err)
//	}
//
//	return result.ErrorOrNil()
//
//}

// train invokes the functions in each train stage and
// returns the total time that the model spent training
func (job *TrainJob) train() (time.Duration, error) {

	var result *multierror.Error

	job.logger.Info("Started new epoch",
		zap.Int("epoch", job.epoch))
	startTime := time.Now()

	// Invoke the functions and get the ids of the functions
	// that should be used to update the model
	funcs := job.invokeTrainFunctions()
	// Get the elapsed time
	elapsed := time.Now().Sub(startTime)

	// Update the model
	//err := job.updateModel(funcs...)
	//result = multierror.Append(result, err)

	// Merge the results from the functions
	job.optimizer.Merge(job.model, funcs...)
	// update the model and invoke the functions
	err := job.model.Save()
	result = multierror.Append(result, err)

	job.logger.Info("Epoch finished, saving model")

	return elapsed, result.ErrorOrNil()
}

// validate invokes the validation function
func (job *TrainJob) validate() {
	// Invoke the validation function while we wait for the scheduler
	job.wgVal.Add(1)
	go job.invokeValFunction(job.wgVal)
}
