package main

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/thesis/ml/pkg/model"
	"go.uber.org/zap"
)

func PanicIf(err error){
	if err != nil {
		panic(err)
	}

}

func main() {

	client := redisai.Connect(fmt.Sprintf("redis://%s:%d", "192.168.99.102", 6379), nil)
	defer client.Close()
	fmt.Println(client.Pool.IdleTimeout)

	l, err := zap.NewDevelopment()
	PanicIf(err)

	// Get the values of the gradients
	m := model.NewModel(l,"example","resnet", []string{"conv1", "conv2", "fc1", "fc2"}, 0.01, client)

	l.Info("Building model...")
	err = m.Build()
	PanicIf(err)


	l.Info("Updating with gradients")
	err = m.Update( "1")
	PanicIf(err)

	l.Info("Saving the new model to the database")
	err = m.Save()


	//con, err := redis.DialURL("redis://192.168.99.102:6379")
	//PanicIf(err)
	//
	//dt, shape, values, err := client.TensorGetValues("example:fc1-weight")
	//fmt.Println(shape, dt)
	//
	//args := redis.Args{}
	//args = args.Add("other", "FLOAT").AddFlat(shape)
	//fmt.Println(args)
	//
	//fc1 := new(bytes.Buffer)
	//binary.Write(fc1, binary.LittleEndian, values.([]float32))
	//
	//args = args.Add("BLOB").Add(fc1.Bytes())
	////fmt.Println(args)
	//
	////args = args.Add("BLOB").AddFlat(values)
	////fmt.Println(args)
	//// TODO need to do this like this if we want it to work
	//_, err = con.Do("AI.TENSORSET", args...)
	//"example", "FLOAT", 128, 9216,"VALUES", m.Layers[3].BiasShape,m.Layers[3].Bias.Data()

	PanicIf(err)






}
