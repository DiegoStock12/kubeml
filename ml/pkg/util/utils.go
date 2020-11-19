package util

import (
	"net"
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
