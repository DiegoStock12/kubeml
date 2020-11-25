package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
)


const (
	routerUrl = "http://192.168.99.101:32422"
	functionName = "network"
)

func panicIf(err error) {
	if err != nil {
		panic(err)
	}
}


func main() {


	values := url.Values{}
	values.Set("task", "init")
	values.Set("psId", "example")
	values.Set("psPort", "34523")
	values.Set("N", "10")
	values.Set("funcId", strconv.Itoa(1))

	final := routerUrl + "/" + functionName + "?" + values.Encode()
	fmt.Println(final)

	resp, err := http.Get(final)
	panicIf(err)


	body, err := ioutil.ReadAll(resp.Body)
	fmt.Println(string(body))



}