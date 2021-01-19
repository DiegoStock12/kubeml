package ps

import (
	"encoding/json"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/gorilla/mux"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"
)


// updateTask Handles the responses from the scheduler to the
// requests by the parameter servers to
func (ps *ParameterServer) updateTask(w http.ResponseWriter, r *http.Request) {

	vars := mux.Vars(r)
	jobId := vars["jobId"]
	ch, exists := ps.jobIndex[jobId]
	if !exists {
		ps.logger.Error("Received response for non-existing job",
			zap.String("id", jobId),
			zap.Any("index", ps.jobIndex))
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	var resp api.TrainTask
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		ps.logger.Error("Could not read response body",
			zap.Error(err))
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	err = json.Unmarshal(body, &resp)
	if err != nil {
		ps.logger.Error("Could not unmarshal the response json",
			zap.String("request", string(body)),
			zap.Error(err))
		w.WriteHeader(http.StatusBadRequest)
		return
	}


	// Send the response to the channel so the job can
	// update the settings and start the next epoch
	ch <- &resp

}

// startTask Handles the request of the scheduler to create a
// new training job. It creates a new parameter server thread and returns the id
// of the created parameeter server
func (ps *ParameterServer) startTask(w http.ResponseWriter, r *http.Request) {

	var task api.TrainTask
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		ps.logger.Error("Could not read response body",
			zap.Error(err))
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	err = json.Unmarshal(body, &task)
	if err != nil {
		ps.logger.Error("Could not unmarshal the task json",
			zap.String("request", string(body)),
			zap.Error(err))
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	// Create a channel per job which will be used
	// to communicate the new levels of parallelism chosen by
	// the scheduler in coming epochs
	//
	// Also update the number of running tasks in the metrics endpoint
	ch := make(chan *api.TrainTask)
	ps.jobIndex[task.JobId] = ch

	job := newTrainJob(ps.logger, &task, ch, ps.doneChan, ps.scheduler)
	go job.serveTrainJob()
	ps.taskStarted(TrainTask)


	w.WriteHeader(http.StatusOK)
}



// Handle Kubernetes heartbeats
func (ps *ParameterServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

// GetHandler Returns the handler for calls from the functions
func (ps *ParameterServer) GetHandler() http.Handler {
	r := mux.NewRouter()
	r.HandleFunc("/start", ps.startTask).Methods("POST")
	r.HandleFunc("/update/{jobId}", ps.updateTask).Methods("POST")
	r.HandleFunc("/health",ps.handleHealth).Methods("GET")

	return r
}

// Start the API at the given port
// All of the parameter server threads share the same API, and
// they communicate through channels
func (ps *ParameterServer) Serve(port int) {

	ps.logger.Info("Starting Parameter Server api",
		zap.Int("port", port))

	addr := fmt.Sprintf(":%v", port)

	err := http.ListenAndServe(addr, ps.GetHandler())
	ps.logger.Fatal("Parameter Server API done",
		zap.Error(err))
}
