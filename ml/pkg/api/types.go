package api

// Types used by the APIs of the controller and the scheduler

type (

	// TrainRequest is sent to the controller api to start a new training job
	TrainRequest struct {
		ModelType    string  `json:"model_type"`
		BatchSize    int     `json:"batch_size"`
		Epochs       int     `json:"epochs"`
		Dataset      string  `json:"dataset"`
		LearningRate float32 `json:"lr"`
		FunctionName string  `json:"function_name"`
	}

	// TrainTask is sent from the scheduler to the PS
	// with the parallelism needed for the job
	// TODO could use this both ways
	TrainTask struct {
		Parameters  TrainRequest `json:"request"`
		Parallelism int          `json:"parallelism"`
	}

	// JobData is returned by the APIs when scheduling a job
	JobData struct {
		Id string `json:"id"`
	}

	// InferRequest is sent when wanting to get a result back from a trained network
	InferRequest struct {
		ModelId string      `json:"model_id"`
		Data    []Datapoint `json:"data"`
	}

	// A single datapoint plus label
	Datapoint struct {
		Features []float32 `json:"features"`
		Label    float32   `json:"label"`
	}

	// History is the train and validation history of a
	// specific training job
	History struct {
		Id   string               `json:"_id"`
		Data map[string][]float32 `json:"data"`
	}

	// Messages exchanged between the components of the
	// system. Namely Scheduler and PS
	ScheduleRequest struct {
		JobId        string       `json:"job_id"`
		Parameters  TrainRequest `json:"request"`
		ElapsedTime float64      `json:"elapsed_time,omitempty"`
		Parallelism int          `json:"parallelism"`
	}

	ScheduleResponse struct {
		NewParallelism int `json:"new_parallelism"`
	}
)
