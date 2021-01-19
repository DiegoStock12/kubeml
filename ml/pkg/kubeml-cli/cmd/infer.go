package cmd

import (
	"encoding/json"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	controllerClient "github.com/diegostock12/thesis/ml/pkg/controller/client"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"io/ioutil"
)

var (
	// network ID and data where
	// the datapoints are saved in JSON format
	network  string
	dataFile string

	inferCmd = &cobra.Command{
		Use:   "infer",
		Short: "Create an inference task for KubeML",
		RunE:  infer,
	}
)

// infer Creates and submits an inference task
func infer(_ *cobra.Command, _ []string) error {
	controller := controllerClient.MakeClient()
	// TODO should read the file and extract the datapoints

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

	req := api.InferRequest{
		ModelId: network,
		Data:    data,
	}

	resp, err := controller.Infer(&req)
	if err != nil {
		return errors.Wrap(err, "could not complete inference")
	}

	fmt.Println(string(resp))
	return nil
}

func init() {
	rootCmd.AddCommand(inferCmd)

	inferCmd.Flags().StringVarP(&network, "network", "n", "", "Network ID (required)")
	inferCmd.Flags().StringVar(&dataFile, "datafile", "", "File with the data (required)")
	inferCmd.MarkFlagRequired("network")
	inferCmd.MarkFlagRequired("datafile")
}
