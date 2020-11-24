module github.com/diegostock12/thesis/ml

go 1.12

require (
	github.com/RedisAI/redisai-go v1.0.1
	github.com/docopt/docopt-go v0.0.0-20180111231733-ee0de3bc6815
	github.com/fission/fission v1.11.2
	github.com/gomodule/redigo v2.0.0+incompatible
	github.com/google/uuid v1.0.0
	github.com/gorilla/mux v1.8.0
	go.uber.org/zap v1.9.1
	gorgonia.org/tensor v0.9.14
	k8s.io/apimachinery v0.0.0-20190612205821-1799e75a0719
	k8s.io/client-go v12.0.0+incompatible
)

replace gonum.org/v1/gonum v0.7.0 => gonum.org/v1/gonum v0.6.2
