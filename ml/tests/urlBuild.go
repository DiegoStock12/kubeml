package main

import (
	"encoding/json"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"net/url"
	"strconv"
)


func main() {

	baseURL := "http://router.fission"
	function := "example"
	values := url.Values{}
	values.Set("task", "train")
	values.Set("funcId", strconv.Itoa(1))

	final := baseURL + "/" + function + "?" + values.Encode()
	fmt.Println(final)

	// Try to encode a request
	req := &api.TrainRequest{
		ModelType:    "resnet",
		BatchSize:    5,
		Epochs:       5,
		Dataset:      "MNIST",
		LearningRate: 0.01,
		FunctionName: "example",
	}

	body, _ := json.Marshal(req)
	fmt.Printf("%v", string(body))


}