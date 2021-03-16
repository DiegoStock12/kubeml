package main

import (
	"fmt"
	"github.com/gomodule/redigo/redis"
)

func pif(err error){
	if err != nil {
		panic(err)
	}
}

func main()  {

	//logger, _ := zap.NewDevelopment()

	//client := redisai.Connect(fmt.Sprintf("redis://%s:%d", api.RedisAddressDebug, api.RedisPortDebug), nil)

	args := redis.Args{}
	args = args.Add("hola", "Float")
	fmt.Println(args[0])

	//m := model.NewModel(logger, "b8df46ec", api.TrainRequest{
	//	ModelType:    "resnet",
	//	BatchSize:    128,
	//	Epochs:       5,
	//	Dataset:      "mnist",
	//	LearningRate: 0.01,
	//	FunctionName: "network",
	//}, []string{"conv1.weight", "conv2.weight", "fc1.weight", "fc2.weight"}, client )
	//
	//err := m.Build()
	//pif(err)
	//
	//m.Summary()

	//err = m.Update("1")
	//pif(err)
	//
	//err = m.Save()
	//pif(err)

}