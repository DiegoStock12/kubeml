package main

import (
	"fmt"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/diegostock12/kubeml/ml/pkg/controller"
	"github.com/diegostock12/kubeml/ml/pkg/ps"
	"github.com/diegostock12/kubeml/ml/pkg/scheduler"
	"go.uber.org/zap"
	"log"
)

func main(){

	logger, err := zap.NewDevelopment()
	if err != nil {
		log.Fatal("Error building zap logger")
	}



	psUrl := fmt.Sprintf("http://localhost:%d", api.PS_DEBUG_PORT)
	schedulerUrl := fmt.Sprintf("http://localhost:%d", api.SCHEDULER_DEBUG_PORT)

	// Create the scheduler which will trigger the parameter server for now
	// The paramater server will also fetch the layers from the redis db and build a model
	go controller.Start(logger, api.CONTROLLER_DEBUG_PORT ,schedulerUrl)
	go scheduler.Start(logger, api.SCHEDULER_DEBUG_PORT, psUrl)
	go ps.Start(logger, api.PS_DEBUG_PORT, schedulerUrl, false)

	select{}

}
