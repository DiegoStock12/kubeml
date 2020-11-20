package model

import (
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
)

func shapeToIntArray(shape64 ...int64)  []int {
	shape := make([]int, len(shape64))
	for i, d := range shape64 {
		shape[i] = int(d)
	}

	return shape
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
