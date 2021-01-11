package cmd

import (
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	controllerClient "github.com/diegostock12/thesis/ml/pkg/controller/client"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"
)

var (
	network string
	data []interface{}
	dataFile string

	inferCmd = &cobra.Command{
		Use: "infer",
		Short: "Create an inference task for KubeML",
		RunE: infer,
	}
)

// infer Creates and submits an inference task
func infer(_ *cobra.Command, _ []string) error {
	controller := controllerClient.MakeClient()
	// TODO should read the file and extract the datapoints

	req := api.InferRequest{
		ModelId: network,
		Data:    nil,
	}

	resp, err := controller.Infer(&req)
	if err != nil {
		return errors.Wrap(err, "could not complete inference")
	}

	fmt.Println(resp)
	return nil
}



func init() {
	rootCmd.AddCommand(inferCmd)

	inferCmd.Flags().StringVarP(&network, "network", "n", "", "Network ID (required)")
	inferCmd.Flags().StringVar(&dataFile, "dataFile","", "File with the data (required)")
	inferCmd.MarkFlagRequired("network")
	inferCmd.MarkFlagRequired("dataFile")
}
