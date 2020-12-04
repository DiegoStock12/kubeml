package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"go.uber.org/zap"
	"gorgonia.org/tensor"
	"log"
)

const (
	host = "192.168.99.101"
	port = 31618
	testTensorName = "b8df46ec:conv1-bias"
)

func convertShape(dims ...int64) []int {
	shape := make([]int, len(dims))
	for i, d := range dims {
		shape[i] = int(d)
	}

	return shape
}

func getLen(dims ...int64) int64 {
	cum := int64(1)
	for _, v := range dims {
		cum *= v
	}
	return cum
}

func main()  {

	logger, _ := zap.NewDevelopment()

	// Create a client
	client := redisai.Connect(fmt.Sprintf("redis://%s:%d", host, port), nil)

	logger.Debug("loading...")
	// get the test tensor
	_, shape, blob, err := client.TensorGetBlob(testTensorName)
	if err != nil{
		log.Fatal(err)
	}
	fmt.Println(shape)
	logger.Debug("loaded")
	len := getLen(shape...)
	fmt.Println(len)

	//fmt.Println(blob)
	//fmt.Println("datatype is", dt, "and shape is", shape)
	//fmt.Println("values are", values)
	values := make([]float32, len)
	//fmt.Println(values)
	r := bytes.NewReader(blob)
	//fmt.Println(r.Len())
	err = binary.Read(r,binary.LittleEndian, &values)
	if err != nil {
		log.Fatal(err)
	}

	//fmt.Println(values)
	////b := tensor.New(tensor.WithShape(2,2), tensor.WithBacking([]int{1,2,3,4}))
	////fmt.Printf("b:\n%1.1f\n", b)
	//
	//err = binary.Read(bytes.NewReader(blob), binary.LittleEndian, &values)
	//if err != nil {
	//	log.Fatal(err)
	//}
	//fmt.Println(values)
	//
	dim := convertShape(shape...)
	fmt.Println(dim)

	// This works! We are able to work with Tensors that we read from the database
	n := tensor.New(tensor.WithShape(dim...), tensor.WithBacking(values))
	fmt.Printf("%v\n", n)
	fmt.Println(n.Shape())


	// We can also get normal redis entries like this
	//repl, _ := redis.String(client.ActiveConn.Do("GET", "example"))
	//fmt.Println(repl)


}