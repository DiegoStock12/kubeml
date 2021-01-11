package cmd

import (
	"github.com/spf13/cobra"
)

var (
	rootCmd = &cobra.Command{
		Use:   "kubeml",
		Short: "CLI tool for interacting with KubeML",
	}
)

// Execute executes the root command
func Execute() error {
	return rootCmd.Execute()
}
