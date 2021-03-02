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

// finishNotification is received by the merger
// to know which functions to take into account
type finishNotification struct {
	funcId   int
	respChan chan MergeResult
}

type MergeResult int

const (
	MergeSucceeded MergeResult = iota
	MergeFailed
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

	job.schedulerCh <- &state
}

// nextIteration receives updates from the functions, and waits for all of the
// functions to complete the current iteration,
func (job *TrainJob) nextIteration(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	funcId, _ := strconv.Atoi(vars["funcId"])

	// communicate that this function has finished and wait for the
	// merger to respond once finished
	respChan := make(chan MergeResult, 1)
	job.finishCh <- &finishNotification{funcId, respChan}
	job.wgIteration.Done()
	result := <-respChan

	switch result {
	case MergeSucceeded:
		job.logger.Debug("Continuing with next iteration", zap.Int("funcId", funcId))
		w.WriteHeader(http.StatusOK)
		return

	case MergeFailed:
		job.logger.Debug("merge failed, critical failure")
		http.Error(w, "error merging model", http.StatusInternalServerError)
		return
	}

}

func (job *TrainJob) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func (job *TrainJob) GetHandler() http.Handler {
	r := mux.NewRouter()
	r.HandleFunc("/startMerger", job.startTask).Methods("POST")
	r.HandleFunc("/update", job.updateTask).Methods("POST")
	r.HandleFunc("/next/{funcId}", job.nextIteration).Methods("POST")
	r.HandleFunc("/health", job.handleHealth).Methods("GET")
	return r
}

// serveMergeModel starts the routine in charge of receiving the requests for merging the model,
// it merges
func (job *TrainJob) serveMergeModel() {

	for {
		errChan := <-job.startMerger

		for {
			job.logger.Debug("Waiting for functions to finish...")
			job.wgIteration.Wait()

			// get the function ids that will be taken into account
			// when fetching and merging the model
			var funcs []int
			var channels []chan MergeResult
			close(job.finishCh)
			for msg := range job.finishCh {
				funcs = append(funcs, msg.funcId)
				channels = append(channels, msg.respChan)
			}

			// once all are done, merge the model and update
			job.logger.Debug("Merging models after iteration", zap.Ints("finishCh", funcs))
			job.optimizer.Merge(job.model, funcs...)
			err := job.model.Save()
			if err != nil {
				job.logger.Error("error saving model", zap.Error(err))
				for _, ch := range channels {
					if ch != nil {
						ch <- MergeFailed
					}
				}
				errChan <- err
				break
			}

			// initialize the wait group again by checking the number of finished functions
			remaining := job.parallelism - int(job.finishedFuncs)
			if remaining == 0 {

				job.logger.Debug("all functions finished, quiting...")
				break

			} else {
				// reset the wait group and reopen the channel with a buffer
				// size equal to the number of finishCh
				job.wgIteration.Add(remaining)
				job.finishCh = make(chan *finishNotification, remaining)

				// answer to all the non-nil channels
				// a channel is nil if the functions is completely finished
				// it might be that some functions have to do 1 more iteration,
				// so those send a nil channel
				for _, ch := range channels {
					if ch != nil {
						ch <- MergeSucceeded
					}
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
