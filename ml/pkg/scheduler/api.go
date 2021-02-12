package scheduler

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/util"
	"github.com/gorilla/mux"
	"github.com/pkg/errors"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"

	"github.com/diegostock12/thesis/ml/pkg/api"
)

// buildFunctionURL returns the url that the PS will invoke to execute the function
// TODO make this more elegant by not having to add all the parameters
func buildFunctionURL(funcId, numFunc int, task, funcName, psId string) string {

	var routerAddr string
	if util.IsDebugEnv() {
		routerAddr = api.ROUTER_ADDRESS_DEBUG
	} else {
		routerAddr = api.ROUTER_ADDRESS
	}

	values := url.Values{}
	values.Set("task", task)
	values.Set("jobId", psId)
	values.Set("N", strconv.Itoa(numFunc))
	values.Set("funcId", strconv.Itoa(funcId))
	values.Set("batchSize", "0")
	values.Set("lr", "1")

	dest := routerAddr + "/" + funcName + "?" + values.Encode()

	return dest
}

// newParallelism listens to the TrainJobs of the Parameter Server and their
// requests for a new level of parallelism.
// To make this doable we need to put the request in a queue and wait for the scheduler
// to get it and schedule it
func (s *Scheduler) newParallelism(w http.ResponseWriter, r *http.Request) {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		s.logger.Error("Failed to get the training request from the TrainJob", zap.Error(err))
		http.Error(w, "Failed to read request", http.StatusInternalServerError)
		return
	}

	// Receive the train task from the job
	// and insert it in the queue to be scheduled
	var task api.TrainTask
	err = json.Unmarshal(body, &task)
	if err != nil {
		s.logger.Error("Failed to parse the trainjob request",
			zap.Error(err),
			zap.String("payload", string(body)))
		http.Error(w, "Failed to decode the request", http.StatusInternalServerError)
		return
	}

	s.logger.Debug("Received request for new parallelism",
		zap.Any("task", task))

	// Add the request to scheduler queue
	s.queue.pushTask(&task)

	w.WriteHeader(http.StatusOK)

}

// train receives the TrainRequest from the controller and enqueues them
func (s *Scheduler) train(w http.ResponseWriter, r *http.Request) {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		s.logger.Error("Failed to get the training request from the API", zap.Error(err))
		http.Error(w, "Failed to read request", http.StatusInternalServerError)
		return
	}

	var req api.TrainRequest
	err = json.Unmarshal(body, &req)
	if err != nil {
		s.logger.Error("Failed to parse the train request",
			zap.Error(err),
			zap.String("payload", string(body)))
		http.Error(w, "Failed to decode the request", http.StatusInternalServerError)
		return
	}

	// Create the jobId and push to queue
	id := createJobId()

	// TODO now add it directly to the task queue
	task := api.TrainTask{
		Parameters: req,
		Job: api.JobInfo{
			JobId: id,
		},
	}

	s.logger.Debug("Adding task to queue",
		zap.Any("task", task))
	s.queue.pushTask(&task)

	s.logger.Debug("here")

	w.WriteHeader(http.StatusOK)
	_, err = w.Write([]byte(id))
	if err != nil {
		s.logger.Error("error writing response", zap.Error(err))
	}
}

// Handle requests to infer with some datapoints
func (s *Scheduler) infer(w http.ResponseWriter, r *http.Request) {
	// For now handle all the inference requests directly without a queue
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		s.logger.Error("Could not unpack infer request", zap.Error(err))
		http.Error(w, errors.Wrap(err, "could not read request").Error(), http.StatusInternalServerError)
		return
	}

	var req api.InferRequest
	// read the train request
	err = json.Unmarshal(body, &req)
	if err != nil {
		s.logger.Error("Failed to parse the train request",
			zap.Error(err),
			zap.String("payload", string(body)))
		http.Error(w, "Failed to decode the request", http.StatusInternalServerError)
		return
	}

	// TODO funcName could be model id
	url := buildFunctionURL(0, 1, "infer", "network", req.ModelId)
	s.logger.Debug("Build inference url", zap.String("url", url))

	resp, err := http.Post(url, "application/json", bytes.NewBuffer(body))
	if err != nil {
		s.logger.Error("Could not receive function response", zap.Error(err))
		http.Error(w, "Failed to receive function response", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	preds, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		s.logger.Error("Could not parse predictions", zap.Error(err))
		http.Error(w, "Failed to unpack predictions", http.StatusInternalServerError)
		return
	}

	s.logger.Debug("got response", zap.String("predictions", string(preds)))
	w.WriteHeader(http.StatusOK)
	w.Header().Set("Content-Type", "application/json")
	w.Write(preds)
}

// taskFinished simply deletes the entry from the scheduler index
func (s *Scheduler) taskFinished(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	taskId := vars["taskId"]

	s.logger.Debug("Deleting task from the cache",
		zap.String("task", taskId))

	s.policy.taskFinished(taskId)

	w.WriteHeader(http.StatusOK)
	return
}

// Handle heartbeats from Kubernetes
func (s *Scheduler) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

// Create the handler for the scheduler to receive requests from the API
func (s *Scheduler) GetHandler() http.Handler {
	r := mux.NewRouter()
	r.HandleFunc("/job", s.newParallelism).Methods("POST")
	r.HandleFunc("/train", s.train).Methods("POST")
	r.HandleFunc("/infer", s.infer).Methods("POST")
	r.HandleFunc("/health", s.handleHealth).Methods("GET")
	r.HandleFunc("/finish/{taskId}", s.taskFinished).Methods("DELETE")
	return r
}

// Expose the API
func (s *Scheduler) Serve(port int) {
	s.logger.Info("Starting scheduler api", zap.Int("port", port))
	addr := fmt.Sprintf(":%v", port)

	// Train serving the endpoint
	err := http.ListenAndServe(addr, s.GetHandler())
	s.logger.Fatal("Scheduler API done", zap.Error(err))

}
