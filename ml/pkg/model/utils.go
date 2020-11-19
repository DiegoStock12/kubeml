package model

import "fmt"

func shapeToIntArray(shape64 ...int64)  []int {
	shape := make([]int, len(shape64))
	for i, d := range shape64 {
		shape[i] = int(d)
	}

	return shape
}

func getWeightKeys(layerName string, grad bool) (string, string){

	var weightName, biasName string

	if grad {
		// Get the name of the gradients according to the layerName
		weightName = fmt.Sprintf("%s%s%s", layerName, weightSuffix, gradientSuffix)
		biasName = fmt.Sprintf("%s%s%s", layerName, biasSuffix, gradientSuffix)
	} else {
		weightName = fmt.Sprintf("%s%s", layerName, weightSuffix)
		biasName = fmt.Sprintf("%s%s", layerName, biasSuffix)
	}

	return weightName, biasName
}
