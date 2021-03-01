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

func main() {

	config := zap.NewDevelopmentConfig()
	config.DisableStacktrace = true
	//config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder

	logger, err := config.Build()
	if err != nil {
		log.Fatal("Error building zap logger")
	}

	psUrl := fmt.Sprintf("http://localhost:%d", api.ParameterServerPortDebug)
	schedulerUrl := fmt.Sprintf("http://localhost:%d", api.SchedulerPortDebug)

	// Create the scheduler which will trigger the parameter server for now
	// The paramater server will also fetch the layers from the redis db and build a model
	go controller.Start(logger, api.ControllerPortDebug, schedulerUrl, psUrl)
	go scheduler.Start(logger, api.SchedulerPortDebug, psUrl)
	go ps.Start(logger, api.ParameterServerPortDebug, schedulerUrl, false)

	select {}

}
