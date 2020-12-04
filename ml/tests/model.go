package main

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/diegostock12/thesis/ml/pkg/model"
	"go.uber.org/zap"
)

func pif(err error){
	if err != nil {
		panic(err)
	}
}

func main()  {

	logger, _ := zap.NewDevelopment()

	client := redisai.Connect(fmt.Sprintf("redis://%s:%d", api.REDIS_ADDRESS_DEBUG, api.REDIS_PORT_DEBUG), nil)

	m := model.NewModel(logger, "b8df46ec", api.TrainRequest{
		ModelType:    "resnet",
		BatchSize:    128,
		Epochs:       5,
		Dataset:      "mnist",
		LearningRate: 0.01,
		FunctionName: "network",
	}, []string{"conv1", "conv2", "fc1", "fc2"}, client )

	err := m.Build()
	pif(err)

	m.Summary()

	err = m.Update("1")
	pif(err)

	err = m.Save()
	pif(err)

}