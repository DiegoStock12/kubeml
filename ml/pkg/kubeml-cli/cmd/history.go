package cmd

import (
	"fmt"
	controllerClient "github.com/diegostock12/thesis/ml/pkg/controller/client"
	"github.com/spf13/cobra"
	"os"
	"text/tabwriter"
)

var (
	taskId     string

	historyCmd = &cobra.Command{
		Use:   "history",
		Short: "Check training history for task",
	}

	historyGetCmd = &cobra.Command{
		Use:   "get",
		Short: "Check training history for task",
		RunE:  getHistory,
	}

	historyDeleteCmd = &cobra.Command{
		Use:   "delete",
		Short: "Check training history for task",
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
	controller := controllerClient.MakeClient()

	history, err := controller.GetHistory(taskId)
	if err != nil {
		return err
	}
	fmt.Println(history)
	return nil
}

// deleteHistory deletes a history from the database given the taskId
func deleteHistory(_ *cobra.Command, _ []string) error {
	controller := controllerClient.MakeClient()

	err := controller.DeleteHistory(taskId)
	if err != nil {
		return err
	}

	fmt.Println("History deleted")
	return nil
}

func listHistories(_ *cobra.Command, _ []string) error {
	controller := controllerClient.MakeClient()

	histories, err := controller.ListHistories()
	if err != nil {
		return err
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 1, ' ', 0)
	fmt.Fprintf(w, "%v\t%v\t%v\t%v\t%v\t%v\n", "NAME", "MODEL", "DATASET", "EPOCHS", "BATCH SIZE", "LR")

	for _, h := range histories {
		fmt.Fprintf(w, "%v\t%v\t%v\t%v\t%v\t%v\n",
			h.Id, h.Task.ModelType, h.Task.Dataset, h.Task.Epochs, h.Task.BatchSize, h.Task.LearningRate)
	}

	w.Flush()

	return nil
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
