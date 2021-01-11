package cmd

import (
	controllerClient "github.com/diegostock12/thesis/ml/pkg/controller/client"
	"github.com/spf13/cobra"
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
)

// createDataset creates a dataset in KubeML
func createDataset(_ *cobra.Command, _ []string) error {
	controller := controllerClient.MakeClient()

	// pass the commands to the client creation command
	return controller.CreateDataset(name, trainData, trainLabels, testData, testLabels)
}

// deleteDataset deletes a dataset from KubeML
func deleteDataset(_ *cobra.Command, _ []string) error {
	controller := controllerClient.MakeClient()

	// return the deletion
	return controller.DeleteDataset(name)
}

func init() {
	rootCmd.AddCommand(datasetCmd)
	datasetCmd.AddCommand(datasetCreateCmd, datasetDeleteCmd)

	// Add the flags to each command
	// Flags for the create command
	datasetCreateCmd.Flags().StringVarP(&name, "name", "n", "", "Dataset Name (required)")
	datasetCreateCmd.Flags().StringVar(&trainData, "trainData", "", "Path to train data (required)")
	datasetCreateCmd.Flags().StringVar(&trainLabels, "trainLabels", "", "Path to train labels (required)")
	datasetCreateCmd.Flags().StringVar(&testData, "testData", "", "Path to test data (required")
	datasetCreateCmd.Flags().StringVar(&testLabels, "testLabels", "", "Path to test labels (required)")

	// Mark all of them as required
	datasetCreateCmd.MarkFlagRequired("name")
	datasetCreateCmd.MarkFlagRequired("trainData")
	datasetCreateCmd.MarkFlagRequired("trainLabels")
	datasetCreateCmd.MarkFlagRequired("testData")
	datasetCreateCmd.MarkFlagRequired("testLabels")

	// Flags for the delete command
	datasetDeleteCmd.Flags().StringVarP(&name, "name", "n", "", "Dataset Name (required)")
	datasetDeleteCmd.MarkFlagRequired("name")
}
