module github.com/diegostock12/thesis/ml

go 1.12

require (
	github.com/RedisAI/redisai-go v1.0.1
	github.com/docopt/docopt-go v0.0.0-20180111231733-ee0de3bc6815
	github.com/gomodule/redigo v2.0.0+incompatible
	github.com/google/uuid v1.1.1
	github.com/gorilla/mux v1.8.0
	github.com/hashicorp/errwrap v0.0.0-20180715044906-d6c0cd880357 // indirect
	github.com/hashicorp/go-multierror v0.0.0-20180717150148-3d5d8f294aa0
	github.com/pkg/errors v0.9.1
	github.com/xdg/stringprep v1.0.0 // indirect
	go.mongodb.org/mongo-driver v1.4.3
	go.uber.org/atomic v1.3.2 // indirect
	go.uber.org/multierr v1.1.0 // indirect
	go.uber.org/zap v1.9.1
	gonum.org/v1/netlib v0.0.0-20190331212654-76723241ea4e // indirect
	gorgonia.org/tensor v0.9.14
)

replace gonum.org/v1/gonum v0.7.0 => gonum.org/v1/gonum v0.6.2
