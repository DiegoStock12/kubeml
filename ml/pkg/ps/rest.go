package ps

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"go.uber.org/zap"
	"net/http"
	"time"
)

// sendSchedulerRequest notifies the scheduler about the completion of an epoch and sends the
// basic information such as elapsed time, number of functions and so on
func (job *TrainJob) sendSchedulerRequest(elapsed time.Duration) {

	addr := fmt.Sprintf("%s:%d%s", api.DEBUG_URL, api.SCHEDULER_DEBUG_PORT, api.SCHEDULER_JOB_ENDPOINT)
	job.logger.Debug("Built response address", zap.String("url", addr))

	job.task.ElapsedTime = elapsed.Seconds()

	body, err := json.Marshal(job.task)
	if err != nil {
		job.logger.Error("Could not marshal trainjob update",
			zap.Error(err),
			zap.Any("request", job.task))
		return
	}

	_, err = http.Post(addr, "application/json", bytes.NewBuffer(body))
	if err != nil {
		job.logger.Error("Could not send request to scheduler",
			zap.Error(err),
			zap.Any("response", job.task))
		return
	}

}
