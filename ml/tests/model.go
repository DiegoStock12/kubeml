package main

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/diegostock12/kubeml/ml/pkg/model"
	"go.uber.org/zap"
	"reflect"
)


const (
	ip = "35.222.29.140"
	port = 6379
)

func pif(err error){
	if err != nil {
		panic(err)
	}
}

func main()  {

	logger, _ := zap.NewDevelopment()

	client := redisai.Connect(fmt.Sprintf("redis://%s:%d", ip, port), nil)



	m := model.NewModel(logger, "test", api.TrainRequest{
		ModelType:    "resnet",
		BatchSize:    32,
		Epochs:       5,
		Dataset:      "mnist",
		LearningRate: 0.01,
		FunctionName: "network",
	}, []string{"test", "test2"}, client )

	err := m.Build()
	pif(err)

	m.Summary()

	fmt.Println(reflect.TypeOf(m.StateDict["test"].Weights.Data()))
	fmt.Println(reflect.TypeOf(m.StateDict["test2"].Weights.Data()))

	//fmt.Println(m.StateDict["test"].Dtype == redisai.TypeInt64)
	//
	err = m.Save()
	pif(err)

}