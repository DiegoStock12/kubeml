package ps

import (
	"context"
	"errors"
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/diegostock12/thesis/ml/pkg/model"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.uber.org/zap"
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
		schedChan <-chan *api.ScheduleResponse
		// epochChan is to synchronize when receiving all the responses from
		// the functions
		// TODO we can just wait until all the functions are ready with a WG
		epochChan chan struct{}

		// Communicate with the cache
		redisClient *redisai.Client

		// request which has to be satisfied
		task *api.TrainTask

		// history of the train job
		history map[string][]float32
	}
)

// newTrainJob Creates a new TrainJob that will take care of a specific train request
func newTrainJob(logger *zap.Logger, id string,
	task *api.TrainTask, schedChan <-chan *api.ScheduleResponse) *TrainJob {

	logger.Info("Creating new train job")

	// Create the connection to the REDIS api that we'll pass through to the PS
	client := redisai.Connect(fmt.Sprintf("redis://%s:%d", api.REDIS_ADDRESS_DEBUG, api.REDIS_PORT_DEBUG), nil)

	// Create the PS struct
	job := &TrainJob{
		logger:      logger.Named(fmt.Sprintf("trainJob-%s", id)),
		jobId:       id,
		parallelism: task.Parallelism,
		epoch:       1,
		schedChan:   schedChan,
		redisClient: client,
		task:        task,
		history:     make(map[string][]float32),
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
	for job.epoch <= job.task.Parameters.Epochs {

		job.logger.Info("Started new epoch",
			zap.Int("epoch", job.epoch))

		// Invoke the functions
		// The function itself launches multiple
		// requests and waits until all of them are done, so
		// we can just block until all of them are fully completed
		// TODO we could do the thing of adding an extra b funcs to deal with stragglers
		job.invokeTrainFunctions()

		// The model updates and so on is handled in parallel in the API
		// Here we just wait for all functions to be done
		//<-job.epochChan

		job.logger.Info("Epoch finished, saving model")
		// update the model and invoke the functions
		err := job.model.Save()
		if err != nil {
			job.logger.Error("Could not update model",
				zap.Error(err))
		}

		// TODO handle the response from the val func
		// Invoke the validation function while we wait for the scheduler
		go job.invokeValFunction()

		// TODO send request to the scheduler through the API

		job.logger.Debug("Waiting for scheduler response")
		resp := <-job.schedChan

		job.logger.Info("Received next config from the Scheduler",
			zap.Int("new parallelism", resp.NewParallelism))

		if resp.NewParallelism < 1 {
			job.logger.Error("Received bad configuration from the scheduler",
				zap.Int("parallelism", resp.NewParallelism))
		}

		// Change the new limits
		job.parallelism = resp.NewParallelism

		// Increment the epoch
		job.epoch++
	}

	job.logger.Info(fmt.Sprintf("Training finished after %d epochs", job.epoch))

	// TODO should save results of the training in the database
	job.saveTrainingHistory()
	job.logger.Info("Exiting...")

}

// saveTrainingHistory saves the history in the mongo database
func (job *TrainJob) saveTrainingHistory() {
	// get the mongo connection
	client, err := mongo.NewClient(options.Client().ApplyURI(createMongoURI()))
	if err != nil {
		job.logger.Error("Could not create mongo client", zap.Error(err))
		return
	}

	// Save the history in the kubeml database in the history collections
	err = client.Connect(context.TODO())
	if err != nil {
		job.logger.Error("Could not connect to mongo", zap.Error(err))
		return
	}

	// Create the history and index by id
	collection := client.Database("kubeml").Collection("history")
	h := api.History{
		Id:   job.jobId,
		Data: job.history,
	}

	// insert it in the DB
	resp, err := collection.InsertOne(context.TODO(), h)
	if err != nil {
		job.logger.Error("Could not insert the history in the database",
			zap.Error(err))
	}

	job.logger.Info("Inserted history", zap.Any("id", resp.InsertedID))

}
