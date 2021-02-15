package main

import (
	"net/http"
)

func main() {

	//err := errors.New("this is an example error")
	//errbyte := []byte(err.Error())
	_, e := http.Post("http://localhost:9999/test", "text/plain", nil)
	if e != nil {
		panic(e)
	}

}
