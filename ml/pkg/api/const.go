package api

// Addresses of services
const (
	FissionRouterUrl   = "http://router.fission"
	StorageUrl         = "http://storage.kubeml"
	SchedulerUrl       = "http://scheduler.kubeml"
	ParameterServerUrl = "http://parameter-server.kubeml"
	ControllerUrl      = "http://controller.kubeml"
	MongoUrl           = "mongodb.kubeml"
	MongoPort          = 27017
	RedisUrl           = "redisai.kubeml"
	RedisPort          = 6379
)

const DefaultParallelism = 5

// Debug
const (
	MongoUrlDebug            = "mongodb://192.168.99.101:30074"
	StorageAddressDebug      = "http://192.168.99.102:9090"
	FissionRouterUrlDebug    = "http://192.168.99.101:32422"
	RedisAddressDebug        = "192.168.99.101"
	RedisPortDebug           = 30358
	DebugParallelism         = 2
	SchedulerPortDebug       = 10200
	ParameterServerPortDebug = 10300
	ControllerPortDebug      = 10100
	HostUrlDebug             = "http://localhost"
)
