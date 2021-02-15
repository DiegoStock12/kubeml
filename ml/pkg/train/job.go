package train

import (
	"errors"
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/diegostock12/kubeml/ml/pkg/model"
	psClient "github.com/diegostock12/kubeml/ml/pkg/ps/client"
	schedulerClient "github.com/diegostock12/kubeml/ml/pkg/scheduler/client"
	"github.com/diegostock12/kubeml/ml/pkg/util"
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

		// history gathers the progress of the job
		// it holds metrics for validation, train loss,
		// accuracy, parallelism and epoch duration
		// for every epoch
		// It is saved to the database after the training process
		// is complete
		history api.JobHistory

		// client for the scheduler (shared by all trainjobs)
		scheduler *schedulerClient.Client
		ps        *psClient.Client

		// request which has to be satisfied
		task        *api.TrainTask
		jobId       string
		parallelism int
		epoch       int

		// reference model
		model     *model.Model
		optimizer model.ParallelSGD

		// channels for communicating with the scheduler
		// and parameter server to get new tasks and send finish
		// signal
		//TODO get rid of the done chan and make all through the api
		schedChan   chan *api.JobState
		doneChan    chan<- string
		redisClient *redisai.Client

		// wait group used when launching a validation
		// function so we do not accidentally exit the job without
		wgVal *sync.WaitGroup
	}
)

// NewTrainJob Creates a new TrainJob that will take care of a specific train request
func NewTrainJob(
	logger *zap.Logger,
	task *api.TrainTask,
	schedChan chan *api.JobState,
	doneChan chan string,
	client *schedulerClient.Client) *TrainJob {

	logger.Info("Creating new train job")

	var redisClient *redisai.Client
	if util.IsDebugEnv() {
		redisClient = redisai.Connect(fmt.Sprintf(
			"redis://%s:%d", api.REDIS_ADDRESS_DEBUG, api.REDIS_PORT_DEBUG), nil)
	} else {
		redisClient = redisai.Connect(fmt.Sprintf("redis://%s:%d", api.REDIS_ADDRESS, api.REDIS_PORT), nil)
	}

	job := &TrainJob{
		logger:      logger.Named(fmt.Sprintf("trainJob-%s", task.Job.JobId)),
		scheduler:   client,
		jobId:       task.Job.JobId,
		parallelism: task.Job.State.Parallelism,
		epoch:       1,
		schedChan:   schedChan,
		doneChan:    doneChan,
		redisClient: redisClient,
		task:        task,
		history:     api.JobHistory{},
		wgVal:       &sync.WaitGroup{},
	}

	var psUrl string
	if util.IsDebugEnv() {
		psUrl = fmt.Sprintf("http://localhost:%v", api.PS_DEBUG_PORT)
	} else {
		psUrl = api.PARAMETER_SERVER_URL
	}
	job.ps = psClient.MakeClient(job.logger, psUrl)
	job.optimizer = model.MakeParallelSGD(job.logger)

	return job

}

// NewBasicJob creates a job with no task provided yet. It will start the job api and
// wait for its task to be defined there.
//
// This is the constructor used when deploying the jobs in separate pods
func NewBasicJob(logger *zap.Logger, jobId string) *TrainJob {
	logger.Info("Creating new basic train job")

	var redisClient *redisai.Client
	if util.IsDebugEnv() {
		redisClient = redisai.Connect(fmt.Sprintf(
			"redis://%s:%d", api.REDIS_ADDRESS_DEBUG, api.REDIS_PORT_DEBUG), nil)
	} else {
		redisClient = redisai.Connect(fmt.Sprintf("redis://%s:%d", api.REDIS_ADDRESS, api.REDIS_PORT), nil)
	}

	job := &TrainJob{
		logger:      logger.Named(fmt.Sprintf("trainJob-%s", jobId)),
		jobId:       jobId,
		epoch:       1,
		schedChan:   make(chan *api.JobState),
		redisClient: redisClient,
		history:     api.JobHistory{},
		wgVal:       &sync.WaitGroup{},
	}

	job.scheduler = schedulerClient.MakeClient(job.logger, api.SCHEDULER_URL)
	job.ps = psClient.MakeClient(job.logger, api.PARAMETER_SERVER_URL)
	job.optimizer = model.MakeParallelSGD(job.logger)

	return job
}

// Train is the main
//
// Waits for the API to receive all the requests for starting the next epoch
// After this the job needs to send a request to the scheduler to get the proper
// amount of functions to use in the next epoch
func (job *TrainJob) Train() {

	job.logger.Info("Starting to serve train job")
	job.logger.Info("Initializing model")

	defer func() {
		// After the job is finished
		// unregister the prometheus exposed metrics,
		// clear connections and send the finish signal to the parameter
		// server
		job.clearTensors()
		job.redisClient.Close()
		job.ps.JobFinished(job.jobId)
	}()

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
		//job.validate()

		job.history.Parallelism = append(job.history.Parallelism, float64(job.parallelism))
		job.history.EpochDuration = append(job.history.EpochDuration, elapsed.Seconds())
		err = job.ps.UpdateMetrics(job.jobId, getLatestMetrics(&job.history))
		if err != nil {
			job.logger.Error("error sending updated metrics to the parameter server",
				zap.Error(err))
		}

		if job.epoch < job.task.Parameters.Epochs {

			job.task.Job.State.ElapsedTime = elapsed.Seconds()
			err = job.scheduler.UpdateJob(job.task)
			if err != nil {
				job.logger.Error("Error updating parallelism",
					zap.Error(err))
				continue
			}

			update := <-job.schedChan
			job.logger.Info("Received next config from the Scheduler",
				zap.Int("new parallelism", update.Parallelism))
			if update.Parallelism < api.DEBUG_PARALLELISM {
				job.logger.Error("Received bad configuration from the scheduler",
					zap.Int("parallelism", update.Parallelism))
			}

			// Get the new parallelism and update it in the history
			// TODO right now in debug environment keep parallelism untouched
			job.task.Job.State = *update
			if !util.IsDebugEnv() && !util.LimitParallelism(){
				job.logger.Debug("updating parallelism...")
				job.parallelism = update.Parallelism
			}

		}
	}

	job.validate()

	// Wait for the val functions to finish
	job.wgVal.Wait()
	job.saveTrainingHistory()

	job.logger.Info("Exiting...", zap.Any("history", job.history))
	job.logger.Info(fmt.Sprintf("Training finished after %d epochs", job.epoch-1))

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

	err = m.Build()
	if err != nil {
		return err
	}

	// Summary of the model
	m.Summary()
	return nil
}

// train invokes the functions in each train stage and
// returns the total time that the model spent training
func (job *TrainJob) train() (time.Duration, error) {

	job.logger.Info("Started new epoch", zap.Int("epoch", job.epoch))

	start := time.Now()
	funcs := job.invokeTrainFunctions()
	elapsed := time.Since(start)

	// Merge the models and update in the database
	job.optimizer.Merge(job.model, funcs...)
	err := job.model.Save()
	job.logger.Info("Epoch finished, saving model")

	return elapsed, err
}

// validate invokes the validation function
func (job *TrainJob) validate() {
	job.wgVal.Add(1)
	go job.invokeValFunction(job.wgVal)
}
