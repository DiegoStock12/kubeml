package cmd

import (
	"github.com/spf13/cobra"
)

var (
	 taskId string
	 outputFile string


	 historyCmd = &cobra.Command{
	 	Use: "history",
	 	Short: "Check training history for task",
	 	RunE: getHistory,
	 }
)

// getHistory gets a training history based on the taskId and pretty
// prints it for easy reference
func getHistory(_ *cobra.Command, _ []string) error {

	// TODO finish this
	return nil

}

func init()  {
	rootCmd.AddCommand(historyCmd)

	historyCmd.Flags().StringVar(&taskId, "taskId", "", "Id of the train task (required)" )
	historyCmd.Flags().StringVarP(&outputFile, "outputFile", "o", "", "Output file to save the results")

	historyCmd.MarkFlagRequired("taskId")
}

