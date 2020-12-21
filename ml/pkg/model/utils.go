package model

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/gomodule/redigo/redis"
	"gorgonia.org/tensor"
)

func shapeToIntArray(shape64 ...int64)  []int {
	shape := make([]int, len(shape64))
	for i, d := range shape64 {
		shape[i] = int(d)
	}

	return shape
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
func blobToFloatArray (blob []byte, shape []int64) ([]float32, error) {
	// Get the total number of components of the tensor
	length := dimsToLength(shape...)
	// allocate the slice
	values := make([]float32, length)

	// read the blob and extract it to the slice
	r := bytes.NewReader(blob)
	err := binary.Read(r, binary.LittleEndian, &values)
	if err != nil { return nil, err}
	return values, nil
}

// fetchTensor abstracts away fetching a tensor from redis in binary format and converting
// it to a tensor. Returns the dimensions and the values of the tensor
func fetchTensor(client *redisai.Client, name string) ([]int64, []float32, error) {
	// Get the tensor from redis
	_, shape, blob, err := client.TensorGetBlob(name)
	if err != nil {return nil, nil, err}

	// Convert the tensor to []float32
	values, err := blobToFloatArray(blob, shape)
	if err != nil {
		return nil, nil, err}

	return shape, values, nil
}

// REDIS gives an error if the layer is too big, we must save the
// layer as a blob directly
func makeArgs(id, name, suffix string, shape tensor.Shape, values interface{}) (*redis.Args, error){

	// Need to get the blob
	valBlob := new(bytes.Buffer)

	err := binary.Write(valBlob, binary.LittleEndian, values.([]float32))
	if err != nil {
		return nil, err
	}

	// build layer name
	entryName := fmt.Sprintf("%s:%s%s", id, name, suffix)

	// Save the weights and the bias
	args := redis.Args{}
	args = args.Add(entryName, "FLOAT").AddFlat(shape)
	args = args.Add("BLOB").Add(valBlob.Bytes())

	return &args, nil
}

func getWeightKeys(layerName string, grad bool, psId string, funcId int) (string, string){

	var weightName, biasName string
	// If it is a gradient and not the initial model we get one for each function
	// We get to index by functionID
	if grad {
		// Get the name of the gradients according to the layerName
		weightName = fmt.Sprintf("%s:%s%s%s/%d", psId, layerName, api.WeightSuffix, api.GradientSuffix, funcId)
		biasName = fmt.Sprintf("%s:%s%s%s/%d", psId, layerName, api.BiasSuffix, api.GradientSuffix, funcId)
	} else {

		// If we have a function Id is because it is not the init model
		if funcId > 0 {
			weightName = fmt.Sprintf("%s:%s%s/%d", psId, layerName, api.WeightSuffix, funcId)
			biasName = fmt.Sprintf("%s:%s%s/%d", psId, layerName, api.BiasSuffix, funcId)
		} else {
			weightName = fmt.Sprintf("%s:%s%s", psId, layerName, api.WeightSuffix)
			biasName = fmt.Sprintf("%s:%s%s", psId, layerName, api.BiasSuffix)
		}
	}

	return weightName, biasName
}


