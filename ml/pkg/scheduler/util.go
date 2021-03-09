package scheduler

import (
	"github.com/google/uuid"
)

// createJobId Creates an ID for the new trainJob
func createJobId() string {
	return uuid.New().String()[:8]
}
