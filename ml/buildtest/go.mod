module github.com/diegostock/test

go 1.12

require (
	github.com/RedisAI/redisai-go v1.0.1
	github.com/docopt/docopt-go v0.0.0-20180111231733-ee0de3bc6815
	github.com/fission/fission v1.11.2
	github.com/gomodule/redigo v2.0.0+incompatible
	github.com/google/uuid v1.0.0
	github.com/gorilla/mux v1.8.0
	github.com/hashicorp/go-multierror v0.0.0-20180717150148-3d5d8f294aa0
	go.mongodb.org/mongo-driver v1.4.3
	go.uber.org/zap v1.9.1
	gorgonia.org/tensor v0.9.14
)

replace gonum.org/v1/gonum v0.7.0 => gonum.org/v1/gonum v0.6.2
