package ps

import (
	"github.com/diegostock12/kubeml/ml/pkg/api"
	schedulerClient "github.com/diegostock12/kubeml/ml/pkg/scheduler/client"
	jobClient "github.com/diegostock12/kubeml/ml/pkg/train/client"
	"github.com/fission/fission/pkg/crd"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
	"k8s.io/client-go/kubernetes"
	"net/http"
	"sync"
)

const (
	KubeMlNamespace = "kubeml"
	KubeMlContainer = "diegostock12/kubeml"
)

// Parameter server is run in a separate goroutine from the scheduler
// It can communicate with the scheduler through channels
type (

	// ParameterServer is the main parameter server that spins out the worker
	// jobs that serve training jobs.
	// The parameter server saves the index to communicate with each of the jobs.
	// The jobs send requests to the Scheduler when they finish an epoch, and the
	// scheduler responds to the main PS api, which then makes the response reach
	// the correct train job.
	ParameterServer struct {
		logger *zap.Logger

		port int

		// clients for other components
		scheduler  *schedulerClient.Client
		jobClient  *jobClient.Client
		kubeClient *kubernetes.Clientset

		// jobIndex with all the train jobs
		// when receiving a response from the scheduler the
		// api will consult the index and send the response to
		// the appropriate worker
		jobIndex map[string]*api.TrainTask
		mu sync.RWMutex

		// flag to choose deployment mode for jobs,
		// false is goroutines and true is in a pod of their own
		// TODO just for A/B testing, choose best one in future
		deployStandaloneJobs bool
	}
)

func serveMetrics(logger *zap.Logger) {

	logger.Debug("Serving metrics")
	// Expose the prometheus metrics endpoint
	http.Handle("/metrics", promhttp.Handler())
	err := http.ListenAndServe(metricsAddr, nil)

	logger.Fatal("metrics endpoint exited", zap.Error(err))

}



// Start Starts a New parameter server which will execute the tasks
//1) start the new functions
//2) receive the notifications from the PS API about functions that have finished processing
//which will trigger the execution retrieval of gradients and the update of the model
//3) Start the API to get the requests from the functions
func Start(logger *zap.Logger, port int, schedulerUrl string, standaloneJobs bool) {

	// build the PS
	ps := &ParameterServer{
		logger:               logger.Named("ps"),
		port:                 port,
		jobIndex:             make(map[string]*api.TrainTask),
		deployStandaloneJobs: standaloneJobs,
	}

	// set the clients
	ps.scheduler = schedulerClient.MakeClient(ps.logger, schedulerUrl)
	ps.jobClient = jobClient.MakeClient(ps.logger)
	_, kubeClient, _, err := crd.GetKubernetesClient()
	if err != nil {
		logger.Fatal("Unable to create kubernetes client", zap.Error(err))
	}
	ps.kubeClient = kubeClient
	ps.logger.Info("Started new parameter server")


	go serveMetrics(ps.logger)

	// Start the API to receive requests
	ps.Serve(port)
}
