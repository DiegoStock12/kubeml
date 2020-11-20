package main

import (
	"github.com/diegostock12/thesis/ml/pkg/scheduler"
	"go.uber.org/zap"
	"log"
)

func main(){

	logger, err := zap.NewDevelopment()
	if err != nil {
		log.Fatal("Error building zap logger")
	}

	// Create the scheduler which will trigger the parameter server for now
	// The paramater server will also fetch the layers from the redis db and build a model
	scheduler.StartScheduler(logger, 9090)

	select{}

}
