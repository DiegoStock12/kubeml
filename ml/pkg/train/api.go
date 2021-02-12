package train

import (
	"fmt"
	"github.com/gorilla/mux"
	"go.uber.org/zap"
	"net/http"
)

// startTask receives the task description from the parameter server and starts
// the training process
func (job *TrainJob) startTask(w http.ResponseWriter, r *http.Request) {

}

// updateTask receives updates from the scheduler with new parameters such as
// parallelism to be applied in the new epochs
func (job TrainJob) updateTask(w http.ResponseWriter, r *http.Request)  {


}

func (job *TrainJob) handleHealth(w http.ResponseWriter, r *http.Request)  {

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
