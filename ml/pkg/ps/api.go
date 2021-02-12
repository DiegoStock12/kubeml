package ps

import (
	"encoding/json"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/gorilla/mux"
	"go.uber.org/zap"
	"io/ioutil"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"net/http"
)


// updateTask Handles the responses from the scheduler to the
// requests by the parameter servers to
func (ps *ParameterServer) updateTask(w http.ResponseWriter, r *http.Request) {

	vars := mux.Vars(r)
	jobId := vars["jobId"]
	ps.mu.RLock()
	task, exists := ps.jobIndex[jobId]
	ps.mu.RUnlock()
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


	// TODO here check if it is a standalone deployment
	// TODO if so send a request using the client, if not get the channel
	// TODO which can be stored in a task object
	// Send the response to the channel so the job can
	// update the settings and start the next epoch
	// TODO see how this is serialized
	task.Job.Channel <- &resp

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

	if ps.deployStandaloneJobs{
		pod, err := ps.createJobPod(task)
		if err != nil {
			ps.logger.Error("error creating pod",
				zap.Error(err))
			http.Error(w, "unable to create pod for job", http.StatusInternalServerError)
			return
		}

		task.Job.Pod = pod

	}
	// Create a channel per job which will be used
	// to communicate the new levels of parallelism chosen by
	// the scheduler in coming epochs
	//
	// Also update the number of running tasks in the metrics endpoint
	ch := make(chan *api.TrainTask)
	ps.jobIndex[task.Job.JobId] = ch

	job := newTrainJob(ps.logger, &task, ch, ps.doneChan, ps.scheduler)
	go job.serveTrainJob()
	taskStarted(TrainTask)


	w.WriteHeader(http.StatusOK)
}


// updateJobMetrics receives the metric updates posted by the training jobs and updates them
// in the prometheus metrics registry
func (ps *ParameterServer) updateJobMetrics(w http.ResponseWriter, r *http.Request)  {
	vars := mux.Vars(r)
	jobId := vars["jobId"]

	var metrics api.MetricUpdate
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		ps.logger.Error("Could not read response body",
			zap.Error(err))
		http.Error(w, "error reading request body", http.StatusInternalServerError)
		return
	}

	err = json.Unmarshal(body, &metrics)
	if err != nil {
		ps.logger.Error("Could not unmarshal the task json",
			zap.String("request", string(body)),
			zap.Error(err))
		http.Error(w, "error reading json body", http.StatusBadRequest)
		return
	}

	// update the metrics for that job
	ps.logger.Debug("Received metrics from job",
		zap.String("jobId", jobId),
		zap.Any("metrics", metrics))

	updateMetrics(jobId, metrics)
	ps.logger.Debug("metrics updated", zap.String("jobId", jobId))

	w.WriteHeader(http.StatusOK)
}


// jobFinish receives the finish signal from the jobs and takes care of the job cleaning
// process.
//
// 1) Deletes the metrics corresponding to that job
// 2) Communicates the finish to the scheduler so it is also cleaned there
// 3) Deletes the Pod using the kubernetes client
// 4) Deletes the entry in the job index of the parameter server
func (ps *ParameterServer) jobFinish(w http.ResponseWriter, r *http.Request)  {
	vars := mux.Vars(r)
	jobId := vars["jobId"]


	ps.mu.RLock()
	task, exists := ps.jobIndex[jobId]
	ps.mu.RUnlock()
	if !exists {
		ps.logger.Error("Received finish from untracked job",
			zap.String("jobId", jobId))
		http.Error(w, "job not found in index", http.StatusBadRequest)
		return
	}


	// clean the metrics for that job
	clearMetrics(jobId)

	// communicate the scheduler that the job is done
	err := ps.scheduler.FinishJob(jobId)
	if err != nil {
		ps.logger.Error("Error sending finish to scheduler",
			zap.Error(err))
	}

	// delete the pod
	// TODO should we retry or something here
	jobPod := task.Job.Pod
	err = ps.kubeClient.CoreV1().Pods(KubeMlNamespace).Delete(jobPod.Name, &metav1.DeleteOptions{})
	if err != nil {
		ps.logger.Error("error deleting pod",
			zap.String("podName", jobPod.Name),
			zap.String("JobId", jobId),
			zap.Error(err))
	}

	// finally delete the pod from the index
	ps.mu.Lock()
	delete(ps.jobIndex, jobId)
	ps.mu.Unlock()

	ps.logger.Debug("Job finished succesfully", zap.String("jobId", jobId))
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
	r.HandleFunc("/metrics/{jobId}", ps.updateJobMetrics).Methods("POST")
	r.HandleFunc("/finish/{jobId}", ps.jobFinish).Methods("DELETE")
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
