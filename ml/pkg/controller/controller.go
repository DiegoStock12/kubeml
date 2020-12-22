package controller

import (
	schedulerClient "github.com/diegostock12/thesis/ml/pkg/scheduler/client"
	"go.uber.org/zap"
)


// TODO the controller should also take care of creating the functions and so on
// TODO look at the fission cli how they create functions and get the code
type (

	// Main struct of the controller
	Controller struct {
		logger *zap.Logger
		scheduler *schedulerClient.Client
	}
)


// Start starts the controller in the specified port
func Start(logger *zap.Logger, port int, schedulerUrl string)  {

	c := &Controller{
		logger: logger.Named("controller"),
	}

	c.scheduler = schedulerClient.MakeClient(c.logger, schedulerUrl)

	go c.Serve(port)

}

