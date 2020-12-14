package main

import (
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/diegostock12/thesis/ml/pkg/controller"
	"github.com/diegostock12/thesis/ml/pkg/ps"
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
	controller.Start(logger, api.CONTROLLER_DEBUG_PORT)
	scheduler.Start(logger, api.SCHEDULER_DEBUG_PORT)
	ps.Start(logger, api.PS_DEBUG_PORT)

	select{}

}
