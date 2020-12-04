package ps

import (
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
)

func createMongoURI() string {
	return fmt.Sprintf("mongodb://%s:%d", api.MONGO_ADDRESS, api.MONGO_PORT)
}