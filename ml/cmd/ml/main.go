package main

import (
	"github.com/diegostock12/thesis/ml/pkg/scheduler"

	"github.com/docopt/docopt-go"
	"go.uber.org/zap"
	"log"
	"strconv"
)

// Gets the port from the argument string
func getPort(logger *zap.Logger, portArg interface{}) int {
	port, err := strconv.Atoi(portArg.(string))
	if err != nil {
		logger.Fatal("Could not parse port", zap.Error(err))
	}
	return port
}

// Run the controller
// TODO implement controller
func runController(logger *zap.Logger, port int) {

}

// Run the scheduler
func runScheduler(logger *zap.Logger, port int) {
	err := scheduler.StartScheduler(logger, port)
	if err != nil {
		logger.Fatal("Scheduler could not start", zap.Error(err))
	}
}

// TODO implement storage
// Run the storage service that will manage the datasets and the trained models
func runStorage(logger *zap.Logger, port int) {

}

// Main function that will run when starting a new pod on Kubernetes.
// Looking at the function arguments the application run will be either a
// Controller or parameter server node
func main() {

	// Create the usage string that will allow us to also read the command line arguments
	usage := `kubeml: Package of the components for running ML applications on Kubernetes

It starts one of the components:
	
	Controller reads the requests of the users and talks to the scheduler
	
	Scheduler decides how many functions to run at a given moment and also 
	creates the parameter servers that interact with the functions and keeps 
	the reference model of the training

	// TODO model manager to save the models and datasets to persistent storage?

Usage:
	kubeml --controllerPort=<port>
	kubeml --schedulerPort=<port>
	kubeml --storageManagerPort=<port>

Options:
	--controllerPort=<port>			Port that the controller should listen on
	--schedulerPort=<port>			Port that the scheduler should listen on
	--storageManagerPort=<port>		Port that the storage manager should listen on
`

	// build development logger that will be passed down
	logger, err := zap.NewDevelopment()
	if err != nil {
		log.Fatalf("Could not build zap logger: %v", err)
	}

	// parse the arguments
	args, err := docopt.ParseDoc(usage)
	if err != nil {
		log.Fatalf("Could not parse the arguments: %v", err)
	}

	// Invoke a specific function depending on what we want to run
	if args["--controllerport"] != nil {
		port := getPort(logger, args["--controllerPort"])
		runController(logger, port)
	}

	// Run scheduler if it is the passed argument
	if args["--schedulerPort"] != nil {
		port := getPort(logger, args["--schedulerPort"])
		runScheduler(logger, port)
	}

	// Run the storage service
	if args["--storageManagerport"] != nil {
		port := getPort(logger, args["--storageManagerPort"])
		runStorage(logger, port)
	}

	// Just wait forever
	select {}

}
