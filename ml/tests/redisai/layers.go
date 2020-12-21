package main

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/diegostock12/thesis/ml/pkg/model"
	"go.uber.org/zap"
)

func main() {

	logger, _ := zap.NewDevelopment()

	// Connect to the client
	conn := redisai.Connect(fmt.Sprintf("redis://%s:%d", "192.168.99.101", 31618), nil)
	defer conn.Close()


	task := api.TrainRequest{
		ModelType:    "resnet",
		BatchSize:    128,
		Epochs:       1,
		Dataset:      "mnist",
		LearningRate: 0.01,
		FunctionName: "network",
	}

	// build the model
	m := model.NewModel(logger, "example", task, []string{"conv1", "conv2", "fc1", "fc2"}, conn)

	// build the optimizer
	//s := model.SGD{Lr: 0.01}


	err := m.Build()
	if err != nil {
		panic(err)
	}
	w1 := m.StateDict["conv1"].Weights
	fmt.Println(w1)
	n, err := m.StateDict["conv1"].Weights.Add(w1)
	ne, err := n.DivScalar(float32(2), true)
	fmt.Println("New matrix\n",ne)


	//if err != nil {
	//	panic(err)
	//}
	//
	//fmt.Println(m.StateDict["conv2"].Bias)
	//
	////err = m.Update("0")
	//err = s.Step(m, 0, 3)
	//if err != nil {
	//	panic(err)
	//}
	//
	//fmt.Println(m.StateDict["conv2"].Bias)
	//fmt.Println(m.StateDict["fc2"].Bias)
	//
	//fmt.Println("built model")
	//
	//
	//err = m.Save()
	//if err != nil {
	//	panic(err)
	//}


}
