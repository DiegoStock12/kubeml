package cmd

import (
	"encoding/json"
	"fmt"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	kubemlClient "github.com/diegostock12/kubeml/ml/pkg/controller/client"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"io/ioutil"
)

var (
	// network ID and data where
	// the datapoints are saved in JSON format
	network  string
	dataFile string
	function string

	inferCmd = &cobra.Command{
		Use:   "infer",
		Short: "Create an inference task for KubeML",
		RunE:  infer,
	}
)

// infer Creates and submits an inference task
func infer(_ *cobra.Command, _ []string) error {
	client, err := kubemlClient.MakeKubemlClient()
	if err != nil {
		return err
	}

	var data []interface{}
	// read the data from the file
	d, err := ioutil.ReadFile(dataFile)
	if err != nil {
		return errors.Wrap(err, "could not read data file")
	}

	err = json.Unmarshal(d, &data)
	if err != nil {
		return errors.Wrap(err, "could not unmarshal data")
	}

	// create the request, we need the function name to know how to refer to the code
	// from the scheduler, and we need the model id to know how to load the appropriate weights
	// from the model storage
	req := api.InferRequest{
		FunctionName: function,
		ModelId: network,
		Data:    data,
	}

	resp, err := client.V1().Networks().Infer(&req)
	if err != nil {
		return errors.Wrap(err, "could not complete inference")
	}

	fmt.Println(string(resp))
	return nil
}

func init() {
	rootCmd.AddCommand(inferCmd)

	inferCmd.Flags().StringVarP(&function, "function", "f", "", "Function Name (required)")
	inferCmd.Flags().StringVarP(&network, "network", "n", "", "Network ID (required)")
	inferCmd.Flags().StringVar(&dataFile, "datafile", "", "File with the data (required)")
	inferCmd.MarkFlagRequired("network")
	inferCmd.MarkFlagRequired("datafile")
	inferCmd.MarkFlagRequired("function")
}
