package api

// Types used by the APIs of the controller and the scheduler

type (


	// TrainRequest is sent to the controller api to start a new training job
	TrainRequest struct {
		ModelType string `json:"model_type"`
		BatchSize int `json:"batch_size"`
		Epochs int `json:"epochs"`
		Dataset string `json:"dataset"`
		LearningRate float64 `json:"lr"`
		FunctionName string `json:"function_name"`
	}

	// InferRequest is sent when wanting to get a result back from a trained network
	InferRequest struct {
		ModelId string   `json:"model_id"`
		Data []Datapoint `json:"data"`
	}

	// A single datapoint plus label
	Datapoint struct {
		Features []float32 `json:"features"`
		Label float32 `json:"label"`
	}

)

