package main

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"gorgonia.org/tensor"
	"log"
)

const (
	host = "192.168.99.102"
	port = 6379
	testTensorName = "grad-conv1"
)

func convertShape(dims ...int64) []int {
	shape := make([]int, len(dims))
	for i, d := range dims {
		shape[i] = int(d)
	}

	return shape
}

func main()  {

	// Create a client
	client := redisai.Connect(fmt.Sprintf("redis://%s:%d", host, port), nil)

	// get the test tensor
	dt, shape, values, err := client.TensorGetValues(testTensorName)
	if err != nil{
		log.Fatal(err)
	}



	fmt.Println("datatype is", dt, "and shape is", shape)
	fmt.Println("values are", values)

	//b := tensor.New(tensor.WithShape(2,2), tensor.WithBacking([]int{1,2,3,4}))
	//fmt.Printf("b:\n%1.1f\n", b)

	dim := convertShape(shape...)
	fmt.Println(dim)

	// This works! We are able to work with Tensors that we read from the database
	n := tensor.New(tensor.WithShape(dim...), tensor.WithBacking(values))
	fmt.Println("n:\n%1.1f\n", n)


}