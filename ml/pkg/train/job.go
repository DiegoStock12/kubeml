package train

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/diegostock12/kubeml/ml/pkg/model"
	psClient "github.com/diegostock12/kubeml/ml/pkg/ps/client"
	schedulerClient "github.com/diegostock12/kubeml/ml/pkg/scheduler/client"
	"github.com/diegostock12/kubeml/ml/pkg/util"
	"github.com/pkg/errors"
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

		// clients for other components
		scheduler   *schedulerClient.Client
		ps          *psClient.Client
		redisClient *redisai.Client

		// Training-specific resources
		history   api.JobHistory
		task      *api.TrainTask
		jobId     string
		epoch     int
		model     *model.Model
		optimizer model.ParallelSGD

		// options of the trainjob
		parallelism   int
		static        bool
		validateEvery int

		// channels for communicating with the scheduler
		// and parameter server to get new tasks and send finish
		// signal
		schedChan chan *api.JobState

		// wait group used when launching a validation
		// function so we do not accidentally exit the job without
		wgVal *sync.WaitGroup

		// function synchronization, waitgroup
		// and index to track functions during an iteration
		wgIteration *sync.WaitGroup
		funcIndex   functionIndex

		exitErr error
	}
)

// NewTrainJob Creates a new TrainJob that will take care of a specific train request
func NewTrainJob(
	logger *zap.Logger,
	task *api.TrainTask,
	schedChan chan *api.JobState,
	client *schedulerClient.Client) *TrainJob {

	logger.Info("Creating new train job")

	var redisClient *redisai.Client
	if util.IsDebugEnv() {
		redisClient = redisai.Connect(fmt.Sprintf(
			"redis://%s:%d", api.RedisAddressDebug, api.RedisPortDebug), nil)
	} else {
		redisClient = redisai.Connect(fmt.Sprintf("redis://%s:%d", api.RedisUrl, api.RedisPort), nil)
	}

	job := &TrainJob{
		logger:        logger.Named(fmt.Sprintf("trainJob-%s", task.Job.JobId)),
		scheduler:     client,
		jobId:         task.Job.JobId,
		schedChan:     schedChan,
		redisClient:   redisClient,
		task:          task,
		history:       api.JobHistory{},
		wgVal:         &sync.WaitGroup{},
		parallelism:   task.Job.State.Parallelism,
		static:        task.Parameters.Options.StaticParallelism,
		validateEvery: task.Parameters.Options.ValidateEvery,
		wgIteration:   &sync.WaitGroup{},
		funcIndex:     newIndex(task.Job.State.Parallelism),
	}

	var psUrl string
	if util.IsDebugEnv() {
		psUrl = fmt.Sprintf("http://localhost:%v", api.ParameterServerPortDebug)
	} else {
		psUrl = api.ParameterServerUrl
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
			"redis://%s:%d", api.RedisAddressDebug, api.RedisPortDebug), nil)
	} else {
		redisClient = redisai.Connect(fmt.Sprintf("redis://%s:%d", api.RedisUrl, api.RedisPort), nil)
	}

	job := &TrainJob{
		logger:      logger.Named(fmt.Sprintf("trainJob-%s", jobId)),
		jobId:       jobId,
		schedChan:   make(chan *api.JobState),
		redisClient: redisClient,
		history:     api.JobHistory{},
		wgVal:       &sync.WaitGroup{},
	}

	job.scheduler = schedulerClient.MakeClient(job.logger, api.SchedulerUrl)
	job.ps = psClient.MakeClient(job.logger, api.ParameterServerUrl)
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
		job.ps.JobFinished(job.jobId, job.exitErr)
	}()

	// Call the init function and build the reference model,
	// fatal if it fails
	err := job.init()
	if err != nil {
		job.logger.Error("Could not initialize model",
			zap.Error(err))
		job.exitErr = err
		return
	}

	// Main training loop
	for job.epoch = 1; job.epoch <= job.task.Parameters.Epochs; job.epoch++ {

		// call all the training functions,
		// gather the stats and return the time taken in the
		// iteration
		err := job.train()
		if err != nil {
			job.logger.Error("Error training model", zap.Error(err))
			job.exitErr = err
			return
		}

		// Trigger validation if configured
		if job.validateEvery != 0 && job.validateEvery%job.epoch == 0 {
			// Invoke the validation function
			job.validate()
		}

		// If we need, ask the scheduler for updated settings
		if !job.static && job.epoch < job.task.Parameters.Epochs {
			err = job.scheduler.UpdateJob(job.task)
			if err != nil {
				job.logger.Error("Error updating parallelism",
					zap.Error(err))
				continue
			}

			update := <-job.schedChan
			job.logger.Info("Received next config from the Scheduler",
				zap.Int("new parallelism", update.Parallelism))

			// Get the new parallelism and update it in the history
			// TODO right now in debug environment keep parallelism untouched
			job.task.Job.State = *update
			if !util.IsDebugEnv() && !util.LimitParallelism() {
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

// init launches the function and creates the model used by the TrainJob
func (job *TrainJob) init() error {

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

	m.Summary()
	return nil
}

// train invokes the functions in each train stage and
// returns the total time that the model spent training
func (job *TrainJob) train() error {
	job.logger.Info("Started new epoch", zap.Int("epoch", job.epoch))

	start := time.Now()
	loss, funcs, err := job.invokeTrainFunctions()
	if err != nil {
		return errors.Wrap(err, "error invoking functions")
	}

	// update the elapsed time
	elapsed := time.Since(start)
	job.task.Job.State.ElapsedTime = elapsed.Seconds()

	// Merge the models and update in the database
	job.optimizer.Merge(job.model, funcs...)

	job.logger.Info("Epoch finished, saving model")
	err = job.model.Save()
	if err != nil {
		return errors.Wrap(err, "error saving model in the database")
	}

	// update the training metrics
	err = job.updateTrainMetrics(loss, elapsed)
	if err != nil {
		job.logger.Error("error updating metrics", zap.Error(err))
	}

	job.logger.Debug("History updated", zap.Any("history", job.history))
	return nil
}

// validate invokes the validation function
func (job *TrainJob) validate() {
	job.wgVal.Add(1)
	go job.invokeValFunction(job.wgVal)
}
