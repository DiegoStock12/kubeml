package cmd

import (
	"bufio"
	"fmt"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"os"
	"os/exec"
)

var (

	// name of the job that we want the logs from
	jobId  string
	follow bool

	logsCmd = &cobra.Command{
		Use:   "logs",
		Short: "Get the logs from a KubeML job",
		RunE:  getLogs,
	}
)

// getLogs allows to get the logs of a job or also follow the logs
// it does it by calling the kubeml command with the appropriate job id and
// namespace
func getLogs(_ *cobra.Command, _ []string) error {

	jobName := fmt.Sprintf("job-%s", jobId)
	command := "kubectl"
	args := []string{"-n", "kubeml", "logs", jobName}

	// if follow add the f flag to keep the
	// logs open
	if follow {
		args = append(args, "-f")
	}

	// create the command struct
	cmd := exec.Command(command, args...)

	// get the pipe to stdout and start the command
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return errors.Wrap(err, "could not access command stdout")
	}
	cmd.Start()

	// get a buffered reader of the command output. If the follow
	// flag is set, keep reading from there until Ctrl+C is called by the user
	// if not, just copy the reader content to the standard output
	reader := bufio.NewReader(stdout)
	if follow {
		for {
			line, _, _ := reader.ReadLine()
			fmt.Println(string(line))
		}
	} else {
		reader.WriteTo(os.Stdout)
	}

	return nil

}

func init() {
	rootCmd.AddCommand(logsCmd)

	logsCmd.Flags().StringVar(&jobId, "id", "", "Id of the jobs to get the logs from")
	logsCmd.Flags().BoolVarP(&follow, "follow", "f", false, "Whether to follow the output")

	logsCmd.MarkFlagRequired("id")

}
