package cmd

import (
	"errors"
	"fmt"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	kubemlClient "github.com/diegostock12/kubeml/ml/pkg/controller/client"
	"github.com/fission/fission/pkg/crd"
	"github.com/hashicorp/go-multierror"
	"github.com/spf13/cobra"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	maxBatchSize = 1024
)

var (

	// variables used in the train command
	dataset      string
	epochs       int
	batchSize    int
	lr           float32
	functionName string

	trainCmd = &cobra.Command{
		Use:   "train",
		Short: "Create a train task for KubeML",
		RunE:  train,
	}
)

// train builds the request and sends it to the controller so
// the job can be scheduled
func train(_ *cobra.Command, _ []string) error {
	client := kubemlClient.MakeKubemlClient()
	req := api.TrainRequest{
		ModelType:    "example",
		BatchSize:    batchSize,
		Epochs:       epochs,
		Dataset:      dataset,
		LearningRate: lr,
		FunctionName: functionName,
	}

	// validate the train request fields
	if err := validateTrainRequest(&req); err != nil {
		return err
	}

	id, err := client.V1().Networks().Train(&req)
	if err != nil {
		return err
	}

	fmt.Println("Started train job with Id",id)
	return nil

}

// validateTrainRequest checks for the validity of the request parameters
// before submitting it to the controller
func validateTrainRequest(req *api.TrainRequest) error {

	e :=  &multierror.Error{}

	// check appropriate batch size
	if req.BatchSize <= 0 || req.BatchSize > maxBatchSize {
		e = multierror.Append(e, errors.New(fmt.Sprintf("batch size should be between %v and %v", 0, maxBatchSize)))
	}

	// check appropriate epochs
	if epochs <= 0 {
		e = multierror.Append(e, errors.New("epochs should be a positive value"))
	}

	// check learning rate
	if lr <= 0 {
		e = multierror.Append(e, errors.New("learning rate should be bigger than zero"))
	}

	// check dataset exists
	if exists, err := datasetExists(dataset); err != nil || !exists {
		e = multierror.Append(e, fmt.Errorf("dataset %v does not exist", name))
	}

	// check function exists
	if exists, err := functionExists(functionName); err != nil || !exists {
		e = multierror.Append(e, fmt.Errorf("function %v does not exist", functionName))
	}


	return e.ErrorOrNil()
}

// datasetExists returns true if dataset is present in kubeml
func datasetExists(name string) (bool, error) {

	client := kubemlClient.MakeKubemlClient()

	_, err := client.V1().Datasets().Get(name)
	if err != nil {
		return false, err
	}

	return true, nil

}

// functionExists returns true if function is in kubeml
func functionExists(name string) (bool, error) {

	fissionClient, _, _, err := crd.MakeFissionClient()
	if err != nil {
		return false, err
	}

	// check if the fission function exists
	_, err = fissionClient.CoreV1().Functions("").Get(name, metav1.GetOptions{})
	if err == nil {
		return true, nil
	}
	return false, err

}

func init() {
	rootCmd.AddCommand(trainCmd)

	trainCmd.Flags().StringVarP(&dataset, "dataset", "d", "", "Dataset name (required)")
	trainCmd.Flags().StringVarP(&functionName, "function", "f", "", "Function name (required)")
	trainCmd.Flags().IntVarP(&epochs, "epochs", "e", 1, "Number of epochs to run (required)")
	trainCmd.Flags().IntVarP(&batchSize, "batch", "b", 64, "Batch Size (required)")
	trainCmd.Flags().Float32Var(&lr, "lr", 0.01, "Learning Rate (required)")

	trainCmd.MarkFlagRequired("dataset")
	trainCmd.MarkFlagRequired("function")
	trainCmd.MarkFlagRequired("epochs")
	trainCmd.MarkFlagRequired("batch")
	trainCmd.MarkFlagRequired("lr")
}
