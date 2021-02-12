package train

import (
	"encoding/json"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/gorilla/mux"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"
)

// startTask receives the task description from the parameter server and starts
// the training process
func (job *TrainJob) startTask(w http.ResponseWriter, r *http.Request) {

	job.logger.Debug("creating task and starting training...")

	var task api.TrainTask
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		job.logger.Error("Could not read request body",
			zap.Error(err))
		http.Error(w, "could not read request body", http.StatusInternalServerError)
		return
	}

	err = json.Unmarshal(body, &task)
	if err != nil {
		job.logger.Error("Could not unmarshal the task json",
			zap.String("request", string(body)),
			zap.Error(err))
		http.Error(w, "could not unmarshal task", http.StatusBadRequest)
		return
	}

	// assign the task to the job and call start
	job.task = &task
	job.parallelism = task.Job.State.Parallelism

	job.logger.Debug("Assigned new task to the job",
		zap.Any("task", task))

	// TODO have a running boolean to prevent us from starting another task
	go job.Train()

	w.WriteHeader(http.StatusOK)
}

// updateTask receives updates from the scheduler with new parameters such as
// parallelism to be applied in the new epochs
func (job TrainJob) updateTask(w http.ResponseWriter, r *http.Request) {

	job.logger.Debug("Updating task")

	var state api.JobState
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		job.logger.Error("Could not read request body",
			zap.Error(err))
		http.Error(w, "could not read request body", http.StatusInternalServerError)
		return
	}

	err = json.Unmarshal(body, &state)
	if err != nil {
		job.logger.Error("Could not unmarshal the state json",
			zap.String("request", string(body)),
			zap.Error(err))
		http.Error(w, "could not unmarshal task", http.StatusBadRequest)
		return
	}

	job.schedChan <- &state
}

func (job *TrainJob) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func (job *TrainJob) GetHandler() http.Handler {
	r := mux.NewRouter()
	r.HandleFunc("/start", job.startTask).Methods("POST")
	r.HandleFunc("/update", job.updateTask).Methods("POST")
	r.HandleFunc("/health", job.handleHealth).Methods("GET")
	return r
}

func (job *TrainJob) Serve(port int) {

	job.logger.Info("starting job API", zap.String("JobID", job.jobId))
	addr := fmt.Sprintf(":%v", port)

	err := http.ListenAndServe(addr, job.GetHandler())
	job.logger.Fatal("Job api quit",
		zap.Error(err))
}
