package train

import (
	"encoding/json"
	"fmt"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/gorilla/mux"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"
	"strconv"
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

	// initialize variables used during training
	job.task = &task
	job.parallelism = task.Job.State.Parallelism
	job.static = task.Parameters.Options.StaticParallelism
	job.validateEvery = task.Parameters.Options.ValidateEvery
	job.wgIteration.Add(job.parallelism)

	// start the model merging thread
	go job.serveMergeModel()

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

// nextIteration receives updates from the functions, and waits for all of the
// functions to complete the current iteration,
func (job *TrainJob) nextIteration(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	funcId, _ := strconv.Atoi(vars["funcId"])

	// communicate that this function has finished
	job.funcs <- funcId
	job.wgIteration.Done()

	<-job.

}

func (job *TrainJob) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func (job *TrainJob) GetHandler() http.Handler {
	r := mux.NewRouter()
	r.HandleFunc("/start", job.startTask).Methods("POST")
	r.HandleFunc("/update", job.updateTask).Methods("POST")
	r.HandleFunc("/next/{funcId}", job.nextIteration).Methods("POST")
	r.HandleFunc("/health", job.handleHealth).Methods("GET")
	return r
}

// serveMergeModel serves for
func (job *TrainJob) serveMergeModel() {

	for {
		<-job.start

		for {
			job.logger.Debug("Waiting for functions to finish...")
			job.wgIteration.Wait()

			// get the functions that we will add to the merge
			var funcs []int
			close(job.funcs)
			for funcId := range job.funcs {
				funcs = append(funcs, funcId)
			}

			// once all are done, merge the model and update
			job.logger.Debug("Merging models after iteration", zap.Ints("funcs", funcs))
			job.optimizer.Merge(job.model, funcs...)
			err := job.model.Save()
			if err != nil {
				// TODO handle this error nicely
				job.logger.Fatal("error saving model")
			}

			// initialize the waitgroup again by checking the number of finished functions
			remaining := job.parallelism - int(job.runningFuncs)
			if remaining == 0 {
				job.logger.Debug("all functions finished")
				break
			} else {

				// reset the wait group and 
				job.wgIteration.Add(remaining)
				job.funcs = make(chan int, remaining)
				for i := 0; i < remaining; i++ {
					job.funcs <- i
				}
			}
		}
	}

}

func (job *TrainJob) Serve(port int) {

	job.logger.Info("starting job API", zap.String("JobID", job.jobId))
	addr := fmt.Sprintf(":%v", port)

	err := http.ListenAndServe(addr, job.GetHandler())
	job.logger.Fatal("Job api quit",
		zap.Error(err))
}
