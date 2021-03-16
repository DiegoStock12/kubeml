package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/gomodule/redigo/redis"
	"gorgonia.org/tensor"
	"strconv"
	"time"
)

const (
	addr       = "35.225.76.167"
	remotePort = 6379
)

// REDIS gives an error if the layer is too big, we must save the
// layer as a blob directly
func makeArgs(id, name string, shape tensor.Shape, values interface{}) (*redis.Args, error) {

	// Need to get the blob
	valBlob := new(bytes.Buffer)

	err := binary.Write(valBlob, binary.LittleEndian, values.([]float32))
	if err != nil {
		return nil, err
	}

	// build layer name
	entryName := fmt.Sprintf("%s:%s", id, name)

	// Save the weights and the bias
	args := redis.Args{}
	args = args.Add(entryName, "FLOAT").AddFlat(shape)
	args = args.Add("BLOB").Add(valBlob.Bytes())

	return &args, nil
}

// fetchTensor abstracts away fetching a tensor from redis in binary format and converting
// it to a tensor. Returns the dimensions and the values of the tensor
func fetchTensor(client *redisai.Client, name string) ([]int64, []float32, error) {
	// Get the tensor from redis
	_, shape, blob, err := client.TensorGetBlob(name)
	if err != nil {
		return nil, nil, err
	}

	// Convert the tensor to []float32
	values, err := blobToFloatArray(blob, shape)
	if err != nil {
		return nil, nil, err
	}

	return shape, values, nil
}

//dimsToLength to parse a blob to a flatten array of floats we need to build
// a fixed size slice, this we do by taking the dimensions of the tensor and multiplying
// them, so we can allocate a slice of that length onto which unpack the blob
func dimsToLength(dims ...int64) int64 {
	accum := int64(1)
	for _, v := range dims {
		accum *= v
	}
	return accum
}

//blobToFloatArray takes the blob returned by Redis (needed to make the tensor loading
// far faster) and translates into a float array that can then be used to build
// a gorgonia tensor
func blobToFloatArray(blob []byte, shape []int64) ([]float32, error) {
	// Get the total number of components of the tensor
	length := dimsToLength(shape...)
	// allocate the slice
	values := make([]float32, length)

	// read the blob and extract it to the slice
	r := bytes.NewReader(blob)
	err := binary.Read(r, binary.LittleEndian, &values)
	if err != nil {
		return nil, err
	}
	return values, nil
}

func main() {

	//logger, _ := zap.NewDevelopment()
	//
	// Create a client
	client := redisai.Connect(fmt.Sprintf("redis://%s:%d", addr, remotePort), nil)

	t := tensor.New(tensor.WithBacking(tensor.Random(tensor.Float32, 32)), tensor.WithShape(4, 8))
	fmt.Println(t)

	client.Pipeline(6)

	start := time.Now()
	client.DoOrSend("MULTI",  nil, nil)
	for i := 0; i < 20; i++ {
		args, _ := makeArgs("test", strconv.Itoa(i), t.Shape(), t.Data())
		//fmt.Println(args)

		_, err := client.DoOrSend("AI.TENSORSET", *args, nil)
		if err != nil {
			panic(err)
		}

	}
	r, err := client.ActiveConn.Do("EXEC")
	if err != nil {
		panic(err)
	}

	fmt.Println(r, len(r.([]interface{})))






	//args := redis.Args{}.Add(fmt.Sprintf("%v:%v", "test", 1), redisai.TensorContentTypeMeta, redisai.TensorContentTypeBlob)
	//_, err := client.DoOrSend("AI.TENSORGET", args, nil)

	//_, _, _, err := client.TensorGetBlob("test:1")
	//if err != nil {
	//	panic(err)
	//}
	//
	//err = client.Flush()
	//if err != nil {
	//	panic(err)
	//}
	//
	//r, err := client.Receive()
	//if err != nil {
	//	panic(err)
	//}
	//
	//fmt.Println(redisai.ProcessTensorGetReply(r, err))

	// get those values
	for i := 0; i < 20; i++ {
		//fmt.Println(args)

		_, _, _, err := client.TensorGetBlob(fmt.Sprintf("%v:%v", "test", i))
		if err != nil {
			panic(err)
		}

		//fmt.Println(redisai.ProcessTensorGetReply(r, err))
	}

	err = client.Flush()
	if err != nil {
		panic(err)
	}

	for i := 0; i < 20; i++ {
		r, err := client.Receive()
		err, _, shape, blob := redisai.ProcessTensorGetReply(r, err)
		arr, err := blobToFloatArray(blob.([]byte), shape)
		fmt.Println(i, shape, arr)
		//fmt.Println(shape)
	}

	fmt.Println("Took ", time.Since(start).Seconds())

	//client.Pipeline(10)
	//client.

}
