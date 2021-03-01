package api

// Addresses of services
const (
	StorageUrl         = "http://storage.kubeml"
	SchedulerUrl       = "http://scheduler.kubeml"
	ParameterServerUrl = "http://parameter-server.kubeml"
	ControllerUrl      = "http://controller.kubeml"
)

const (
	FissionRouterUrl = "http://router.fission"
	MongoUrl         = "mongodb.kubeml"
	MongoPort        = 27017
	RedisUrl         = "redisai.kubeml"
	RedisPort        = 6379
)

// Debug
const (
	MongoUrlDebug            = "mongodb://192.168.99.101:30933"
	StorageAddressDebug      = "http://192.168.99.102:9090"
	FissionRouterUrlDebug    = "http://192.168.99.101:32422"
	RedisAddressDebug        = "192.168.99.101"
	RedisPortDebug           = 31618
	DebugParallelism         = 2
	SchedulerPortDebug       = 10200
	ParameterServerPortDebug = 10300
	ControllerPortDebug      = 10100
	HostUrlDebug             = "http://localhost"
)
