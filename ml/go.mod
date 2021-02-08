module github.com/diegostock12/thesis/ml

go 1.12

require (
	github.com/RedisAI/redisai-go v1.0.1
	github.com/coreos/go-systemd v0.0.0-20190719114852-fd7a80b32e1f // indirect
	github.com/docopt/docopt-go v0.0.0-20180111231733-ee0de3bc6815
	github.com/fission/fission v1.8.1-0.20210208054438-6f9bad3d05f8
	github.com/golang/groupcache v0.0.0-20200121045136-8c9f03a8e57e // indirect
	github.com/gomodule/redigo v2.0.0+incompatible
	github.com/google/uuid v1.1.2
	github.com/googleapis/gnostic v0.3.0 // indirect
	github.com/gophercloud/gophercloud v0.2.0 // indirect
	github.com/gorilla/mux v1.8.0
	github.com/hashicorp/go-multierror v1.0.0
	github.com/json-iterator/go v1.1.9 // indirect
	github.com/kr/pty v1.1.8 // indirect
	github.com/onsi/ginkgo v1.7.0 // indirect
	github.com/onsi/gomega v1.4.3 // indirect
	github.com/pierrec/lz4 v2.0.5+incompatible // indirect
	github.com/pkg/errors v0.9.1
	github.com/prometheus/client_golang v1.0.0
	github.com/spf13/cobra v1.1.1
	github.com/stretchr/objx v0.2.0 // indirect
	go.mongodb.org/mongo-driver v1.4.3
	go.uber.org/multierr v1.5.0 // indirect
	go.uber.org/zap v1.10.0
	golang.org/x/exp v0.0.0-20200224162631-6cc2880d07d6 // indirect
	golang.org/x/time v0.0.0-20191024005414-555d28b269f0 // indirect
	gopkg.in/inf.v0 v0.9.1 // indirect
	gorgonia.org/tensor v0.9.14
	k8s.io/api v0.0.0-20190620084959-7cf5895f2711
	k8s.io/apimachinery v0.0.0-20190612205821-1799e75a0719
	k8s.io/client-go v12.0.0+incompatible
)

// replace needed because of gonum 0.7.0 used in
// gorgonia incompatible with go 1.12
replace gonum.org/v1/gonum v0.7.0 => gonum.org/v1/gonum v0.6.2
