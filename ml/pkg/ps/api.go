package ps

import (
	"encoding/json"
	"fmt"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/diegostock12/kubeml/ml/pkg/train"
	"github.com/gorilla/mux"
	"go.uber.org/zap"
	"io/ioutil"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"net/http"
)

// listTasks returns a list of the currently running tasks
func (ps *ParameterServer) listTasks(w http.ResponseWriter, r *http.Request) {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	var tasks []*api.TrainTask
	for _, task := range ps.jobIndex {
		tasks = append(tasks, task)
	}

	resp, err := json.Marshal(tasks)
	if err != nil {
		ps.logger.Error("error marshalling tasks", zap.Error(err))
		http.Error(w, "error sending tasks", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	w.Header().Set("Content-Type", "application/json")
	w.Write(resp)

}

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

	var update api.JobState
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		ps.logger.Error("Could not read state body",
			zap.Error(err))
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	err = json.Unmarshal(body, &update)
	if err != nil {
		ps.logger.Error("Could not unmarshal the state json",
			zap.String("request", string(body)),
			zap.Error(err))
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	// send the update through the client if standalone or
	// through the channel if threaded ps
	if ps.deployStandaloneJobs {
		err = ps.jobClient.UpdateTask(task, update)
		if err != nil {
			ps.logger.Error("could not send update to job",
				zap.Error(err))
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
	} else {
		task.Job.Channel <- &update
	}

}

// startTask Handles the request of the scheduler to create a
// new training job. It creates a new parameter server thread and returns the id
// of the created parameeter server
func (ps *ParameterServer) startTask(w http.ResponseWriter, r *http.Request) {

	var task api.TrainTask
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		ps.logger.Error("Could not read request body",
			zap.Error(err))
		http.Error(w, "could not read request body", http.StatusInternalServerError)
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

	ps.logger.Debug("About to create pod")

	// if we are deploying the jobs in different pods
	// create it and add it to the struct
	if ps.deployStandaloneJobs {
		pod, err := ps.createJobPod(task)
		if err != nil {
			ps.logger.Error("error creating pod",
				zap.Error(err))
			http.Error(w, "unable to create pod for job", http.StatusInternalServerError)
			return
		}
		task.Job.Pod = pod

		ps.logger.Debug("assigned pod to task",
			zap.Any("name", pod.Name),
			zap.Any("ip", pod.Status.PodIP),
			zap.Any("phase", pod.Status.Phase))

		ps.logger.Debug("Sending task to job",
			zap.Any("task", task))

		// here we should send the request to start the task using the client
		// TODO here if we are unable we should repeat and if not in the end delete the pod
		err = ps.jobClient.StartTask(&task)
		if err != nil {
			ps.logger.Error("Unable to send the task to the jobClient",
				zap.Error(err))
			http.Error(w, "unable to send task for job", http.StatusInternalServerError)
			return
		}

		ps.logger.Debug("task sent to job")

	} else {
		// if we are deploying them in the same pod, create a channel to communicate
		ch := make(chan *api.JobState)
		task.Job.Channel = ch
		job := train.NewTrainJob(ps.logger, &task, ch, ps.scheduler)
		go job.Train()
	}

	ps.jobIndex[task.Job.JobId] = &task
	taskStarted(TrainTask)

	w.WriteHeader(http.StatusOK)
}

// updateJobMetrics receives the metric updates posted by the training jobs and updates them
// in the prometheus metrics registry
func (ps *ParameterServer) updateJobMetrics(w http.ResponseWriter, r *http.Request) {
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
func (ps *ParameterServer) jobFinish(w http.ResponseWriter, r *http.Request) {
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

	// delete the pod if there are standalone
	if ps.deployStandaloneJobs {
		// TODO should we retry or something here
		jobPod := task.Job.Pod
		err = ps.kubeClient.CoreV1().Pods(KubeMlNamespace).Delete(jobPod.Name, &metav1.DeleteOptions{})
		if err != nil {
			ps.logger.Error("error deleting pod",
				zap.String("podName", jobPod.Name),
				zap.String("JobId", jobId),
				zap.Error(err))
		}
	}

	// finally delete the pod from the index
	ps.mu.Lock()
	delete(ps.jobIndex, jobId)
	ps.mu.Unlock()

	taskFinished(TrainTask)

	// check if the body is not nil, in that case, report the error to notify of a failure
	if r.Body == http.NoBody {
		ps.logger.Info("Job finished successfully", zap.String("jobId", jobId))
	} else {
		errorStr, err := ioutil.ReadAll(r.Body)
		if err != nil {
			ps.logger.Debug("error reading error body", zap.Error(err))
		} else {
			ps.logger.Info("Job finished with error message",
				zap.String("jobId", jobId),
				zap.String("error", string(errorStr)))
		}
	}

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
	r.HandleFunc("/health", ps.handleHealth).Methods("GET")
	r.HandleFunc("/metrics/{jobId}", ps.updateJobMetrics).Methods("POST")
	r.HandleFunc("/finish/{jobId}", ps.jobFinish).Methods("POST")
	r.HandleFunc("/tasks", ps.listTasks).Methods("GET")
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
