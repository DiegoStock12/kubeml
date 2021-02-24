package cmd

import (
	"fmt"
	kubemlClient "github.com/diegostock12/kubeml/ml/pkg/controller/client"
	"github.com/spf13/cobra"
	"os"
	"text/tabwriter"
)

var (
	short bool

	tasksCmd = &cobra.Command{
		Use:   "task",
		Short: "Manage Running tasks",
	}

	tasksListCmd = &cobra.Command{
		Use:   "list",
		Short: "List deployed running tasks",
		RunE:  listTasks,
	}
)

// listFunctions returns a table with the information of the current functions
func listTasks(_ *cobra.Command, _ []string) error {
	// make fission client
	client := kubemlClient.MakeKubemlClient()

	// get the list of functions and print some of their properties to a table
	tasks, err := client.V1().Tasks().List()
	if err != nil {
		return err
	}

	if short {
		for _, task := range tasks {
			fmt.Println(task.Job.JobId)
		}
		return nil
	}

	w := tabwriter.NewWriter(os.Stdout, 1, 1, 2, ' ', 0)
	fmt.Fprintf(w, "%v\t%v\t%v\t%v\t%v\t%v\t%v\n", "NAME", "FUNCTION", "DATASET", "MODEL", "EPOCHS", "BATCH", "LR")

	// Display functions that use the default environment
	for _, task := range tasks {
		fmt.Fprintf(w, "%v\t%v\t%v\t%v\t%v\t%v\t%v\n",
			task.Job.JobId, task.Parameters.FunctionName, task.Parameters.Dataset,
			task.Parameters.ModelType, task.Parameters.Epochs, task.Parameters.BatchSize, task.Parameters.LearningRate)
	}

	w.Flush()

	return nil
}

func init() {
	rootCmd.AddCommand(tasksCmd)
	tasksCmd.AddCommand(tasksListCmd)

	tasksListCmd.Flags().BoolVar(&short, "short", false, "Trigger short format")
}
