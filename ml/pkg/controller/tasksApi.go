package controller

import (
	"github.com/gorilla/mux"
	"go.uber.org/zap"
	"net/http"
)

// listTasks gets the tasks from the ps and simply redirects them
func (c *Controller) listTasks(w http.ResponseWriter, r *http.Request) {
	taskBytes, err := c.ps.ListTasks()
	if err != nil {
		c.logger.Error("error getting tasks from ps", zap.Error(err))
		http.Error(w, "error getting tasks", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	w.Header().Set("Content-Type", "application/json")
	w.Write(taskBytes)
}

func (c *Controller) stopTask(w http.ResponseWriter, r *http.Request)  {
	vars := mux.Vars(r)
	jobId := vars["jobId"]

	err := c.ps.StopTask(jobId)
	if err != nil {
		c.logger.Error("Error stoping task",
			zap.Error(err))
		http.Error(w, "error stopping task", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
}
