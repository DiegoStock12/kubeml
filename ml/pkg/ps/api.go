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

func respondWithSuccess(w http.ResponseWriter, resp []byte) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_, err := w.Write(resp)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
}

// updateTask Handles the responses from the scheduler to the
// requests by the parameter servers to
func (ps *ParameterServer) updateTask(w http.ResponseWriter, r *http.Request) {

	// Get the job that the new response is for
	vars := mux.Vars(r)
	jobId := vars["jobId"]

	// get the channel of the job
	ch, exists := ps.jobIndex[jobId]
	if !exists {
		ps.logger.Error("Received response for non-existing job",
			zap.String("id", jobId),
			zap.Any("index", ps.jobIndex))
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	// Unpack the Response
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

	ps.logger.Debug("Received response from the scheduler, sending to job...",
		zap.Any("resp", resp))

	// Send the response to the channel
	ch <- &resp

}

// startTask Handles the request of the scheduler to create a
// new training job. It creates a new parameter server thread and returns the id
// of the created parameeter server
func (ps *ParameterServer) startTask(w http.ResponseWriter, r *http.Request) {
	ps.logger.Debug("Processing request from the Scheduler")

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

	ps.logger.Debug("Received task from the scheduler",
		zap.Any("task", task))

	ch := make(chan *api.TrainTask)

	// TODO get a default parallelism
	// Create the train job and start serving
	job := newTrainJob(ps.logger, &task, ch, ps.doneChan, ps.scheduler)
	go job.serveTrainJob()

	// Add the channel and the id to the map
	ps.jobIndex[task.JobId] = ch

	w.WriteHeader(http.StatusOK)
}



// Handle Kubernetes heartbeats
func (ps *ParameterServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

// Returns the handler for calls from the functions
func (ps *ParameterServer) GetHandler() http.Handler {
	r := mux.NewRouter()
	//r.HandleFunc("/finish/{funcId}", ps.handleFinish).Methods("POST")
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

	// Start serving the endpoint
	err := http.ListenAndServe(addr, ps.GetHandler())
	ps.logger.Fatal("Parameter Server API done",
		zap.Error(err))
}
