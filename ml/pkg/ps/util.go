package ps

import (
	"context"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/diegostock12/thesis/ml/pkg/util"
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
		Data: job.history.ToMap(),
	}

	// insert it in the DB
	resp, err := collection.InsertOne(context.TODO(), h)
	if err != nil {
		job.logger.Error("Could not insert the history in the database",
			zap.Error(err))
	}

	job.logger.Info("Inserted history", zap.Any("id", resp.InsertedID))

}