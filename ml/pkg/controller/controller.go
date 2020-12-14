package controller

import (
	"go.uber.org/zap"
)


// TODO the controller should also take care of creating the functions and so on
// TODO look at the fission cli how they create functions and get the code
type (

	// Main struct of the controller
	Controller struct {
		logger *zap.Logger


	}
)

// Redirects a request to the scheduler
// TODO this in the future might have to interact with the storage service to retrieve models etc
func (c *Controller) redirectTrainRequest()  {

}

// Redirect an inference task to the scheduler
func (c *Controller) redirectInferenceRequest() {

}

// Notifies the storage manager and formats the dataset properly
// TODO unimplemented
func (c *Controller) uploadDataset() {

}

// Start starts the controller in the specified port
func Start(logger *zap.Logger, port int)  {

	c := &Controller{
		logger: logger.Named("controller"),
	}

	go c.Serve(port)

}

