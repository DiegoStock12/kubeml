package controller

import (
	"context"
	"fmt"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	schedulerClient "github.com/diegostock12/kubeml/ml/pkg/scheduler/client"
	"github.com/diegostock12/kubeml/ml/pkg/util"
	"github.com/pkg/errors"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.uber.org/zap"
	"log"
)

// TODO the controller should also take care of creating the functions and so on
// TODO look at the fission cli how they create functions and get the code
type (

	// Main struct of the controller
	Controller struct {
		logger      *zap.Logger
		scheduler   *schedulerClient.Client
		mongoClient *mongo.Client
	}
)

func getMongoClient() (*mongo.Client, error) {
	var uri string
	if util.IsDebugEnv() {
		uri = api.MONGO_ADDRESS_DEBUG
	} else {
		uri = fmt.Sprintf("mongodb://%s:%d", api.MONGO_ADDRESS, api.MONGO_PORT)
	}

	client, err := mongo.NewClient(options.Client().ApplyURI(uri))
	if err != nil {
		return nil, err
	}

	err = client.Connect(context.Background())
	if err != nil {
		return nil, errors.Wrap(err, "could not connect to the database")
	}

	return client, nil
}

// Start starts the controller in the specified port
func Start(logger *zap.Logger, port int, schedulerUrl string) {

	c := &Controller{
		logger: logger.Named("controller"),
	}

	// Set the scheduler and mongo clients
	c.scheduler = schedulerClient.MakeClient(c.logger, schedulerUrl)
	client, err := getMongoClient()
	if err != nil {
		log.Fatal(err)
	}
	c.mongoClient = client

	c.Serve(port)

}
