package util

import (
	"fmt"
	"net"
	"os"
	"strconv"
)

// Finds a free port in the current machine/container
func FindFreePort() (int, error) {
	listener, err := net.Listen("tcp", ":0")
	if err != nil {
		return 0, err
	}

	port := listener.Addr().(*net.TCPAddr).Port

	err = listener.Close()
	if err != nil {
		return 0, err
	}

	return port, nil
}


func IsDebugEnv () bool {
	d := os.Getenv("DEBUG_ENV")
	fmt.Println("DEBUG_ENV=", d)
	if len(d) == 0 {
		return false
	}

	debug, err := strconv.ParseBool(d)
	if err != nil {
		panic(err)
	}
	return debug
}
