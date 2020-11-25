package storage

import "go.uber.org/zap"

// TODO this should get the datasets and save them to MONGO

type (
	// Manages the storage of the datasets
	StorageManager struct {
		logger *zap.Logger
	}
)

// SaveDataset fetches a dataset and stores it in the Mongo DB so
// the functions can retrieve it from there
func (sm *StorageManager) SaveDataset(name string)  {

}

