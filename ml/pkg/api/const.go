package api

const (

	// Constants to save and retrieve the gradients
	WeightSuffix   = ".weight"
	BiasSuffix     = ".bias"
	GradientSuffix = ".grad"
)

// Addresses of services
const (
	// Address to access the fission router
	STORAGE_ADDRESS = "http://storage.kubeml"
	SCHEDULER_URL = "http://scheduler.kubeml"
	PARAMETER_SERVER_URL = "http://parameter-server.kubeml"
	CONTROLLER_URL = "http://controller.kubeml"
)

const(
	ROUTER_ADDRESS = "http://router.fission"
	MONGO_ADDRESS  = "mongodb.default"
	MONGO_PORT     = 27017
	REDIS_ADDRESS  = "redisai.default"
	REDIS_PORT = 6379
)

// Debug
const (
	MONGO_ADDRESS_DEBUG = "mongodb://192.168.99.101:30933"
	STORAGE_ADDRESS_DEBUG = "http://192.168.99.102:9090"
	ROUTER_ADDRESS_DEBUG = "http://192.168.99.101:32422"
	REDIS_ADDRESS_DEBUG  = "192.168.99.101"
	REDIS_PORT_DEBUG     = 31618
	DEBUG_PARALLELISM    = 2
	SCHEDULER_DEBUG_PORT = 10200
	PS_DEBUG_PORT = 10300
	CONTROLLER_DEBUG_PORT = 10100
	DEBUG_URL = "http://localhost"
)



