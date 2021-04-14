package model

// TrainAlgorithm exposes the interface that any distributed SGD
// variant must implement to be used with kubeml
type TrainAlgorithm interface {

	// Step updates the central model with the
	// different models from the functions
	Step(funcs ...int) error

	// Save publishes the model(s) needed for the next function
	// iteration to the database
	Save(funcs ...int) error

	// PreTrain performs any operation needed by the algorithm prior
	// to starting the next iteration
	PreTrain() error

	// Update applies any updates to the model when a function finishes
	Update(funcId int) error
}
