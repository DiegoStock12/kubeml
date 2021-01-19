package api

// Types used by the APIs of the controller and the scheduler

type (

	// TrainRequest is sent to the controller api to start a new training job
	// This is then embedded in the Train Task that is used by the PS
	TrainRequest struct {
		ModelType    string  `json:"model_type"`
		BatchSize    int     `json:"batch_size"`
		Epochs       int     `json:"epochs"`
		Dataset      string  `json:"dataset"`
		LearningRate float32 `json:"lr"`
		FunctionName string  `json:"function_name"`
	}

	// InferRequest is sent when wanting to get a result back from a trained network
	InferRequest struct {
		ModelId string        `json:"model_id"`
		Data    []interface{} `json:"data"`
	}

	// TrainTask is sent from the scheduler to the PS
	// with the parallelism needed for the job
	TrainTask struct {
		Parameters  TrainRequest `json:"request"`
		Parallelism int          `json:"parallelism"`
		JobId       string       `json:"job_id"`
		ElapsedTime float64      `json:"elapsed_time"`
	}

	// A single datapoint plus label
	Datapoint struct {
		Features []float32 `json:"features"`
	}

	// History is the train and validation history of a
	// specific training job
	History struct {
		Id   string               `bson:"_id" json:"id"`
		Task TrainRequest         `json:"task"`
		Data map[string][]float64 `json:"data"`
	}


)
