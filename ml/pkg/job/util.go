package ps

import (
	"context"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/diegostock12/thesis/ml/pkg/util"
	"github.com/gomodule/redigo/redis"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.uber.org/zap"
)

func createMongoURI() string {
	if util.IsDebugEnv() {
		return api.MONGO_ADDRESS_DEBUG
	} else {
		return fmt.Sprintf("mongodb://%s:%d", api.MONGO_ADDRESS, api.MONGO_PORT)
	}
}

// lastValue simply returns the last value of the array if not empty,
// if empty return 0
func lastValue(arr []float64) float64 {
	if len(arr) == 0 {
		return 0
	}
	return arr[len(arr)-1]
}

// getLatestMetrics gets the last entry of the history and returns a metrics
// object that will be sent to the parameter server api to update the counters
// of the job
func getLatestMetrics(history *api.JobHistory) api.MetricUpdate {
	return api.MetricUpdate{
		ValidationLoss: lastValue(history.ValidationLoss),
		Accuracy:       lastValue(history.Accuracy),
		TrainLoss:      lastValue(history.TrainLoss),
		Parallelism:    lastValue(history.Parallelism),
		EpochDuration:  lastValue(history.EpochDuration),
	}
}

// clearTensors simply drops the keys and values used during training by the
// different functions and keeps only the reference model in the database
// to save space
func (job *TrainJob) clearTensors() {

	filterStr := fmt.Sprintf("%s*/*", job.jobId)
	tensorListArgs := redis.Args{filterStr}
	tensorNames, err := redis.Strings(job.redisClient.DoOrSend("KEYS", tensorListArgs, nil))
	if err != nil {
		job.logger.Error("Error accessing tensors to be deleted", zap.Error(err))
		return
	}

	// delete the temporary tensors in one call
	deleteArgs := redis.Args{}.AddFlat(tensorNames)
	num, err := redis.Int(job.redisClient.DoOrSend("DEL", deleteArgs, nil))
	if err != nil {
		job.logger.Error("Error deleting database tensors", zap.Error(err))
		return
	}
	if num == 0 {
		job.logger.Warn("No tensors with this name found in the database")
	}
	job.logger.Debug("Delete from the database", zap.Int("num tensors", num))
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
	defer client.Disconnect(context.TODO())

	// Create the history and index by id
	collection := client.Database("kubeml").Collection("history")
	h := api.History{
		Id:   job.jobId,
		Task: job.task.Parameters,
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
