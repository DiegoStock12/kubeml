package util

import (
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
	if len(d) == 0 {
		return false
	}

	debug, _ := strconv.ParseBool(d)
	return debug
}
