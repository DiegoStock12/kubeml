package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
)

func main() {

	follow := false
	args := []string{"urlBuild.go", "-n", "100"}

	if follow {
		args = append(args, "-f")
	}

	cmd := exec.Command("tail", args...)

	stdout, _ := cmd.StdoutPipe()
	cmd.Start()

	buf := bufio.NewReader(stdout)
	if follow {
		for {
			line, _, _ := buf.ReadLine()
			fmt.Println(string(line))
		}
	} else {
		buf.WriteTo(os.Stdout)
	}


	cmd.Wait()

}
