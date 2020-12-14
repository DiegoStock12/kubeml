package scheduler

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"go.uber.org/zap"
	"net/http"
)


// Methods used by the scheduler to send requests and responses
// to the other components in the system


// updateTask sends to the PS APi the settings for the next epoch
func (s *Scheduler) updateTask(task *api.TrainTask)  {

	s.logger.Debug("Sending response back to the TJ",
		zap.Any("response", task))

	// Perform the request
	addr := fmt.Sprintf("%s:%d%s/%s", api.DEBUG_URL, api.PS_DEBUG_PORT, api.PS_UPDATE_ENDPOINT ,task.JobId)
	s.logger.Debug("Built response address", zap.String("url", addr))

	body, err := json.Marshal(task)
	if err != nil {
		s.logger.Error("Could not marshal task update",
			zap.Error(err),
			zap.Any("task", task))
		return
	}

	_, err = http.Post(addr, "application/json", bytes.NewBuffer(body))
	if err != nil {
		s.logger.Error("Could not send update to PS",
			zap.Error(err),
			zap.Any("task", task))
		return
	}

}

// startTask sends a new task to the Parameter Server
// task has the fields Parameters, JobId, parallelism and elapsed time
// which is the last time the job completed a task in
func (s *Scheduler) startTask(task *api.TrainTask) error {

	// send request
	body, err := json.Marshal(task)
	if err != nil {
		return err
	}

	// TODO change this requestURL
	reqURL := fmt.Sprintf("%s:%d", api.DEBUG_URL, api.PS_DEBUG_PORT) + api.PS_START_ENDPOINT
	_, err = http.Post(reqURL, "application/json", bytes.NewReader(body))

	return err
}