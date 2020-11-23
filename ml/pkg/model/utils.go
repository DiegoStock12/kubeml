package model

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/gomodule/redigo/redis"
)

func shapeToIntArray(shape64 ...int64)  []int {
	shape := make([]int, len(shape64))
	for i, d := range shape64 {
		shape[i] = int(d)
	}

	return shape
}

// REDIS gives an error if the layer is too big, we must save the
// layer as a blob directly
func makeArgs(name string, shape []int64, values interface{}) (*redis.Args, error){

	// Need to get the blob
	valBlob := new(bytes.Buffer)

	err := binary.Write(valBlob, binary.LittleEndian, values.([]float32))
	if err != nil {
		return nil, err
	}

	// Save the weights and the bias
	args := redis.Args{}
	args = args.Add(name, "FLOAT").AddFlat(shape)
	args = args.Add("BLOB").Add(valBlob.Bytes())

	return &args, nil
}

func getWeightKeys(layerName string, grad bool, psId, funcId string) (string, string){

	var weightName, biasName string
	// If it is a gradient and not the initial model we get one for each function
	// We get to index by functionID
	if grad {
		// Get the name of the gradients according to the layerName
		weightName = fmt.Sprintf("%s:%s%s%s/%s", psId, layerName, api.WeightSuffix, api.GradientSuffix, funcId)
		biasName = fmt.Sprintf("%s:%s%s%s/%s", psId, layerName, api.BiasSuffix, api.GradientSuffix, funcId)
	} else {
		weightName = fmt.Sprintf("%s:%s%s", psId, layerName, api.WeightSuffix)
		biasName = fmt.Sprintf("%s:%s%s", psId, layerName, api.BiasSuffix)
	}

	return weightName, biasName
}


