package cmd

import (
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	controllerClient "github.com/diegostock12/thesis/ml/pkg/controller/client"
	"github.com/spf13/cobra"
)

var (

	// variables used in the train command
	// todo functionName should be created with fission on the go
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
func train(cmd *cobra.Command, args []string) error {
	controller := controllerClient.MakeClient()
	fmt.Println("Building train request and sending to ", controller)
	req := api.TrainRequest{
		ModelType:    "example",
		BatchSize:    batchSize,
		Epochs:       epochs,
		Dataset:      dataset,
		LearningRate: lr,
		FunctionName: functionName,
	}
	fmt.Println("Request", req)
	id, err := controller.Train(&req)
	if err != nil {
		return err
	}

	fmt.Println(id)
	return nil

}

func init() {
	rootCmd.AddCommand(trainCmd)

	trainCmd.Flags().StringVarP(&dataset, "dataset", "d", "", "Dataset name (required)")
	trainCmd.Flags().StringVarP(&functionName, "function", "f", "", "Function name (required)")
	trainCmd.Flags().IntVarP(&epochs, "epochs", "e", 1, "Number of epochs to run (required)")
	trainCmd.Flags().IntVarP(&batchSize, "batchSize", "b", 64, "Batch Size (required)")
	trainCmd.Flags().Float32Var(&lr, "lr", 0.01, "Learning Rate (required)")

	trainCmd.MarkFlagRequired("dataset")
	trainCmd.MarkFlagRequired("function")
	trainCmd.MarkFlagRequired("epochs")
	trainCmd.MarkFlagRequired("batchSize")
	trainCmd.MarkFlagRequired("lr")
}
