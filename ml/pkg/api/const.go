package api

const (

	// Constants to save and retrieve the gradients
	WeightSuffix   = "-weight"
	BiasSuffix     = "-bias"
	GradientSuffix = "-grad"
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

const (
	ROUTER_ADDRESS_DEBUG = "http://192.168.99.101:32422"
	REDIS_ADDRESS_DEBUG  = "192.168.99.101"
	REDIS_PORT_DEBUG     = 31618
	DEBUG_PARALLELISM    = 4
)
