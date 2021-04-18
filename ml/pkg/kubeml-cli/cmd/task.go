package cmd

import (
	"fmt"
	kubemlClient "github.com/diegostock12/kubeml/ml/pkg/controller/client"
	"github.com/fission/fission/pkg/crd"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"os"
	"text/tabwriter"
)

const KubemlNamespace = "kubeml"

var (
	short bool
	id    string

	tasksCmd = &cobra.Command{
		Use:   "task",
		Short: "Manage Running tasks",
	}

	tasksListCmd = &cobra.Command{
		Use:   "list",
		Short: "List deployed running tasks",
		RunE:  listTasks,
	}

	tasksStopCmd = &cobra.Command{
		Use:   "stop",
		Short: "Stop tasks",
		RunE:  stopTask,
	}

	tasksPruneCmd = &cobra.Command{
		Use:   "prune",
		Short: "Prune finished tasks",
		RunE:  pruneTasks,
	}
)

func stopTask(_ *cobra.Command, _ []string) error {
	// make fission client
	client, err := kubemlClient.MakeKubemlClient()
	if err != nil {
		return err
	}

	err = client.V1().Tasks().Stop(id)
	if err != nil {
		return err
	}

	return nil

}

// pruneTasks deletes all the tasks from the namespace that are
// still left after finishing
func pruneTasks(_ *cobra.Command, _ []string) error {

	_, kubeClient, _, err := crd.GetKubernetesClient()
	if err != nil {
		return errors.Wrap(err, "could not get kubernetes client")
	}

	labelSelector := metav1.LabelSelector{
		MatchLabels: map[string]string{
			"svc": "job",
		},
	}
	listOptions := metav1.ListOptions{
		LabelSelector: labelSelector.String(),
	}

	// get a list of the services and the pods that match the labels
	svcs, err := kubeClient.CoreV1().Services(KubemlNamespace).List(listOptions)
	if err != nil {
		return errors.Wrap(err, "could not list services")
	}

	pods, err := kubeClient.CoreV1().Pods(KubemlNamespace).List(listOptions)
	if err != nil {
		return errors.Wrap(err, "could not list pods")
	}

	for _, svc := range svcs.Items {
		fmt.Println("Service", svc.Name)
	}
	for _, pod := range pods.Items {
		fmt.Println("Pod", pod.Name)
	}

	// delete the tasks and services with the label svc=job
	//kubeClient.CoreV1().Services(KubemlNamespace).Delete(metav1.DeleteOptions{})
	return nil
}

// listFunctions returns a table with the information of the current functions
func listTasks(_ *cobra.Command, _ []string) error {
	// make fission client
	client, err := kubemlClient.MakeKubemlClient()
	if err != nil {
		return err
	}

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
	tasksCmd.AddCommand(tasksStopCmd)
	tasksCmd.AddCommand(tasksPruneCmd)

	tasksListCmd.Flags().BoolVar(&short, "short", false, "Trigger short format")

	tasksStopCmd.Flags().StringVar(&id, "id", "", "Id of the task")
	tasksStopCmd.MarkFlagRequired("id")
}
