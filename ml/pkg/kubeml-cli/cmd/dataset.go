package cmd

import (
	"fmt"
	kubemlClient "github.com/diegostock12/kubeml/ml/pkg/controller/client"
	"github.com/spf13/cobra"
	"os"
	"text/tabwriter"
)

var (

	// variables used in the create command
	// All paths to a file
	trainData   string
	testData    string
	trainLabels string
	testLabels  string

	// Variables used by dataset command in general
	name string

	datasetCmd = &cobra.Command{
		Use:   "dataset",
		Short: "Upload or delete a dataset used by kubeml",
	}

	datasetCreateCmd = &cobra.Command{
		Use:   "create",
		Short: "Create a new dataset in KubeML",
		Long: `Given the paths to the dataset files (train data and labels, test data and labels),
upload the files to KubeMl so they can be used in training tasks. Files must be either .npy or .pkl files`,
		RunE: createDataset,
	}

	datasetDeleteCmd = &cobra.Command{
		Use:   "delete",
		Short: "Delete a dataset in KubeML",
		RunE:  deleteDataset,
	}

	listDatasetCmd = &cobra.Command{
		Use:   "list",
		Short: "List dataset information",
		RunE:  listDatasets,
	}
)

// createDataset creates a dataset in KubeML
func createDataset(_ *cobra.Command, _ []string) error {
	client, err  := kubemlClient.MakeKubemlClient()
	if err != nil {
		return err
	}

	// pass the commands to the client creation command
	return client.V1().Datasets().Create(name, trainData, trainLabels, testData, testLabels)
}

// deleteDataset deletes a dataset from KubeML
func deleteDataset(_ *cobra.Command, _ []string) error {
	client, err  := kubemlClient.MakeKubemlClient()
	if err != nil {
		return err
	}

	// return the deletion
	return client.V1().Datasets().Delete(name)
}

// listDatasets lists the datasets from kubeml
func listDatasets(_ *cobra.Command, _ []string) error {
	client, err := kubemlClient.MakeKubemlClient()
	if err != nil {
		return err
	}

	datasets, err := client.V1().Datasets().List()
	if err != nil {
		return err
	}

	w := tabwriter.NewWriter(os.Stdout, 1, 1, 2, ' ', 0)
	fmt.Fprintf(w, "%v\t%v\t%v\n", "NAME", "TRAINSET", "TESTSET")

	for _, d := range datasets {
		fmt.Fprintf(w, "%v\t%v\t%v\n", d.Name, d.TrainSetSize, d.TestSetSize)
	}

	w.Flush()
	return nil
}

func init() {
	rootCmd.AddCommand(datasetCmd)
	datasetCmd.AddCommand(datasetCreateCmd, datasetDeleteCmd, listDatasetCmd)

	// Add the flags to each command
	// Flags for the create command
	datasetCreateCmd.Flags().StringVarP(&name, "name", "n", "", "Dataset Name (required)")
	datasetCreateCmd.Flags().StringVar(&trainData, "traindata", "", "Path to train data (required)")
	datasetCreateCmd.Flags().StringVar(&trainLabels, "trainlabels", "", "Path to train labels (required)")
	datasetCreateCmd.Flags().StringVar(&testData, "testdata", "", "Path to test data (required")
	datasetCreateCmd.Flags().StringVar(&testLabels, "testlabels", "", "Path to test labels (required)")

	// Mark all of them as required
	datasetCreateCmd.MarkFlagRequired("name")
	datasetCreateCmd.MarkFlagRequired("traindata")
	datasetCreateCmd.MarkFlagRequired("trainlabels")
	datasetCreateCmd.MarkFlagRequired("testdata")
	datasetCreateCmd.MarkFlagRequired("testlabels")

	// Flags for the delete command
	datasetDeleteCmd.Flags().StringVarP(&name, "name", "n", "", "Dataset Name (required)")
	datasetDeleteCmd.MarkFlagRequired("name")
}
