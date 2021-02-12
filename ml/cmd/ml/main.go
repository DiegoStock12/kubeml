package main

import (
	"github.com/diegostock12/kubeml/ml/pkg/controller"
	"github.com/diegostock12/kubeml/ml/pkg/ps"
	"github.com/diegostock12/kubeml/ml/pkg/scheduler"
	"github.com/diegostock12/kubeml/ml/pkg/train"
	"os"

	"github.com/docopt/docopt-go"
	"go.uber.org/zap"
	"log"
	"strconv"
)

func getStringArgWithDefault(arg interface{}, defaultValue string) string {
	if arg != nil {
		return arg.(string)
	} else {
		return defaultValue
	}
}

// Gets the port from the argument string
func getPort(logger *zap.Logger, portArg interface{}) int {
	port, err := strconv.Atoi(portArg.(string))
	if err != nil {
		logger.Fatal("Could not parse port", zap.Error(err))
	}
	return port
}

// Run the controller
func runController(logger *zap.Logger, port int, schedulerUrl string) {
	controller.Start(logger, port, schedulerUrl)
	logger.Fatal("Controller exited")

}

// Run the scheduler
func runScheduler(logger *zap.Logger, port int, psUrl string) {
	scheduler.Start(logger, port, psUrl)
	logger.Fatal("Scheduler exited")
}

// Run the parameter server
func runParameterServer(logger *zap.Logger, port int, schedulerUrl string, standalone bool) {
	ps.Start(logger, port, schedulerUrl, standalone)
	logger.Fatal("Parameter Server exited")
}


func runJob(logger *zap.Logger, port int, jobId string){
	job := train.NewBasicJob(logger, jobId)
	job.Serve(port)
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
	kubeml --jobPort=<port> --jobId=<id>
	kubeml --psPort=<port>


Options:
	--controllerPort=<port>			Port that the controller should listen on
	--schedulerPort=<port>			Port that the scheduler should listen on
	--jobPort=<port>				Port that the job should listen on
	--jobId=<id>					Id of the job to be started
	--psPort=<port> 				Port that the parameter server should listen on
`

	// build development logger that will be passed down
	logger, err := zap.NewDevelopment()
	if err != nil {
		log.Fatalf("Could not build zap logger: %v", err)
	}

	// for now set the default urls
	schedulerUrl := "http://scheduler.kubeml"
	psUrl := "http://parameter-server.kubeml"

	// parse the arguments
	args, err := docopt.ParseDoc(usage)
	if err != nil {
		log.Fatalf("Could not parse the arguments: %v", err)
	}

	// Invoke a specific function depending on what we want to run
	if args["--controllerPort"] != nil {
		port := getPort(logger, args["--controllerPort"])
		runController(logger, port, schedulerUrl)
	}

	// Run ps if it is the passed argument
	if args["--psPort"] != nil {

		var sjobs bool
		// get the argument to see if we run standalone jobs
		standaloneJobs := os.Getenv("STANDALONE_JOBS")

		// by default assume that standalone jobs is true
		if len(standaloneJobs) == 0 {
			sjobs = true
		} else {
			sjobs, err = strconv.ParseBool(standaloneJobs)
			if err != nil{
				logger.Fatal("could not parse env variable for jobs", zap.Error(err))
			}
		}

		port := getPort(logger, args["--psPort"])
		runParameterServer(logger, port, schedulerUrl, sjobs)
	}

	// Run scheduler if it is the passed argument
	if args["--schedulerPort"] != nil {
		port := getPort(logger, args["--schedulerPort"])
		runScheduler(logger, port, psUrl)
	}

	// Run a new train job
	if args["--jobPort"] != nil {
		port := getPort(logger, args["--jobPort"])
		var jobId string
		if args["--jobId"] != nil {
			jobId = args["--jobId"].(string)
		} else {
			logger.Fatal("Given jobPort but not jobId")
		}
		runJob(logger, port, jobId)
	}

	// Just wait forever
	select {}

}
