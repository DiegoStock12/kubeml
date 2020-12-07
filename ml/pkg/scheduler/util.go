package scheduler

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"go.uber.org/zap"
	"net/http"
)

// sendJobResponse sends to the PS APi the settings for the next epoch
func (s *Scheduler) sendJobResponse(parallelism int, jobId string)  {

	// Build the train task
	resp := api.ScheduleResponse{
		NewParallelism: parallelism,
	}

	s.logger.Debug("Sending response back to the TJ",
		zap.Any("response", resp))

	// Perform the request
	addr := fmt.Sprintf("%s:%d/update/%s", api.DEBUG_URL, api.PS_DEBUG_PORT, jobId)
	s.logger.Debug("Built response address", zap.String("url", addr))

	body, err := json.Marshal(resp)
	if err != nil {
		s.logger.Error("Could not marshal trainjob update",
			zap.Error(err),
			zap.Any("response", resp))
		return
	}

	_, err = http.Post(addr, "application/json", bytes.NewBuffer(body))
	if err != nil {
		s.logger.Error("Could not send update to PS",
			zap.Error(err),
			zap.Any("response", resp))
		return
	}

}
