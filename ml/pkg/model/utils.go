package model

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/gomodule/redigo/redis"
	"gorgonia.org/tensor"
)

func shapeToIntArray(shape64 ...int64) []int {
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



// blobToArray converts a byte array to an arrayof int64 with the same shape as indicated
func blobtoIntArray(blob []byte, shape []int64)  ([]int64, error) {
	// Get the total number of components of the tensor
	length := dimsToLength(shape...)
	// allocate the slice
	values := make([]int64, length)

	// read the blob and extract it to the slice
	r := bytes.NewReader(blob)
	err := binary.Read(r, binary.LittleEndian, &values)
	if err != nil {
		return nil, err
	}
	return values, nil

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

// REDIS gives an error if the layer is too big, we must save the
// layer as a blob directly
func makeArgs(id, name string, shape tensor.Shape, dtype string, values interface{}) (*redis.Args, error) {

	// Need to get the blob
	valBlob := new(bytes.Buffer)


	// Some layers inside batch normalization can have special mean and variance
	// tracking modules that are not float arrays. To save those appropriately,
	// analyze the type of the values and cast accordingly when copying the tensor
	switch values.(type) {
	case []float32:
		err := binary.Write(valBlob, binary.LittleEndian, values.([]float32))
		if err != nil {
			return nil, err
		}

	case []int64:
		err := binary.Write(valBlob, binary.LittleEndian, values.([]int64))
		if err != nil {
			return nil, err
		}

	case int64:
		err := binary.Write(valBlob, binary.LittleEndian, values.(int64))
		if err != nil {
			return nil, err
		}

	case float32:
		err := binary.Write(valBlob, binary.LittleEndian, values.(float32))
		if err != nil {
			return nil, err
		}

	}



	// build layer name
	entryName := fmt.Sprintf("%s:%s", id, name)

	// Save the weights and the bias
	args := redis.Args{}
	args = args.Add(entryName, dtype).AddFlat(shape)
	args = args.Add("BLOB").Add(valBlob.Bytes())

	return &args, nil
}

// getWeightKeys returns the proper formatted name of the weights and bias for a specific
// parameter server id and function Id
func getWeightKeys(layerName string, jobId string, funcId int) (string) {

	var weightName string

	// If we have a function Id is because it is not the init model
	// When creating the init model or saving the reference model the tags
	// are like `modelId:conv1.weight` however if it is the model resulting from
	// a training function it will be `modelId:conv1.weight/funcId`
	//
	// For times in which we want to load the init or reference model, we pass
	// -1 in the functionId field
	if funcId >= 0 {
		weightName = fmt.Sprintf("%s:%s/%d", jobId, layerName, funcId)
	} else {
		weightName = fmt.Sprintf("%s:%s", jobId, layerName)
	}

	return weightName
}
