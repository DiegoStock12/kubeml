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
	MONGO_ADDRESS = "mongo.default"
	MONGO_PORT = 27017
	REDIS_ADDRESS = "redis.default"
)

