package cmd

import (
	"encoding/json"
	"fmt"
	kubemlClient "github.com/diegostock12/kubeml/ml/pkg/controller/client"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"math"
	"os"
	"text/tabwriter"
)

var (
	taskId string

	historyCmd = &cobra.Command{
		Use:   "history",
		Short: "Check training history for task",
	}

	historyGetCmd = &cobra.Command{
		Use:   "get",
		Short: "Get training history for task",
		RunE:  getHistory,
	}

	historyDeleteCmd = &cobra.Command{
		Use:   "delete",
		Short: "Delete training history for task",
		RunE:  deleteHistory,
	}

	historyListCmd = &cobra.Command{
		Use:   "list",
		Short: "Get list of networks and summary",
		RunE:  listHistories,
	}
)

// getHistory gets a training history based on the taskId and pretty
// prints it for easy reference
func getHistory(_ *cobra.Command, _ []string) error {
	client, err := kubemlClient.MakeKubemlClient()
	if err != nil {
		return err
	}

	history, err := client.V1().Histories().Get(taskId)
	if err != nil {
		return err
	}

	out, err := json.MarshalIndent(history, "", "  ")
	if err != nil {
		return errors.Wrap(err, "could not marshal json")
	}

	fmt.Println(string(out))
	return nil
}

// deleteHistory deletes a history from the database given the taskId
func deleteHistory(_ *cobra.Command, _ []string) error {
	client, err := kubemlClient.MakeKubemlClient()
	if err != nil {
		return err
	}

	err = client.V1().Histories().Delete(taskId)
	if err != nil {
		return err
	}

	fmt.Println("History deleted")
	return nil
}

func last(arr []float64) float64 {
	if len(arr) > 0 {
		return arr[len(arr)-1]
	}
	return math.NaN()
}

func listHistories(_ *cobra.Command, _ []string) error {
	client, err := kubemlClient.MakeKubemlClient()
	if err != nil {
		return err
	}

	histories, err := client.V1().Histories().List()
	if err != nil {
		return err
	}

	w := tabwriter.NewWriter(os.Stdout, 1, 1, 2, ' ', 0)
	fmt.Fprintf(w, "%v\t%v\t%v\t%v\t%v\t%v\t%v\t%v\t%v\t%v\t%v\n", "NAME", "MODEL", "DATASET", "EPOCHS", "BATCH", "LR", "PARALLELISM", "STATIC", "ACCURACY", "LOSS", "TIME (s)")

	for _, h := range histories {
		fmt.Fprintf(w, "%v\t%v\t%v\t%v\t%v\t%v\t%v\t%v\t%v\t%v\t%v\n",
			h.Id, h.Task.ModelType, h.Task.Dataset, h.Task.Epochs, h.Task.BatchSize, h.Task.LearningRate,
			getMeanParallelism(h.Data.Parallelism), h.Task.Options.StaticParallelism,
			last(h.Data.Accuracy), last(h.Data.ValidationLoss), calculateTotalTime(h.Data.EpochDuration...))
	}

	w.Flush()

	return nil
}

func getMeanParallelism(parallelisms []float64) float64 {
	var total float64 = 0
	for _, p := range parallelisms {
		total += p
	}

	return total / float64(len(parallelisms))

}

func calculateTotalTime(epochs ...float64) float64 {
	var total float64
	for _, t := range epochs {
		total += t
	}
	return total

}

func init() {
	rootCmd.AddCommand(historyCmd)
	historyCmd.AddCommand(historyGetCmd)
	historyCmd.AddCommand(historyDeleteCmd)
	historyCmd.AddCommand(historyListCmd)

	// Get command
	historyGetCmd.Flags().StringVar(&taskId, "network", "", "Id of the train task (required)")

	// Delete command
	historyDeleteCmd.Flags().StringVar(&taskId, "network", "", "Id of the train task (required)")

	historyGetCmd.MarkFlagRequired("network")
	historyDeleteCmd.MarkFlagRequired("network")
}
