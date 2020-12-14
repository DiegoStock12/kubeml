package api

const (

	// Constants to save and retrieve the gradients
	WeightSuffix   = ".weight"
	BiasSuffix     = ".bias"
	GradientSuffix = ".grad"
)

// Redis connection params
const (
	RedisHost = "192.168.99.102"
	RedisPort = 6379
)

const (

	// Address to access the fission router
	ROUTER_ADDRESS = "http://router.fission"
	MONGO_ADDRESS  = "mongo.default"
	MONGO_PORT     = 27017
	REDIS_ADDRESS  = "redis.default"
)

// Debug
const (
	ROUTER_ADDRESS_DEBUG = "http://192.168.99.101:32422"
	REDIS_ADDRESS_DEBUG  = "192.168.99.101"
	REDIS_PORT_DEBUG     = 31618
	DEBUG_PARALLELISM    = 2
	SCHEDULER_DEBUG_PORT = 10200
	PS_DEBUG_PORT = 10300
	CONTROLLER_DEBUG_PORT = 10100
	DEBUG_URL = "http://localhost"
)

// port on which the different API's will listen on
const ML_DEFAULT_PORT = 10200


// TODO create a client for each of these
// Constants with the API endpoints
const (
	SCHEDULER_TRAIN_ENDPOINT = "/train"
	SCHEDULER_INFERENCE_ENDPOINT = "/infer"
	SCHEDULER_JOB_ENDPOINT = "/job"

	CONTROLLER_TRAIN_ENDPOINT = "/train"
	CONTROLLER_INFERENCE_ENDPOINT = "/infer"
	CONTROLLER_DATASET_ENDPOINT = "/dataset"

	PS_START_ENDPOINT = "/start"
	PS_UPDATE_ENDPOINT = "/update"
)



