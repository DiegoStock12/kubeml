package main

import (
	"encoding/json"
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
	values.Set("task", "train")
	values.Set("psId", "example")
	values.Set("psPort", "34523")
	values.Set("N", "50")
	values.Set("funcId", strconv.Itoa(0))
	values.Set("batchSize", strconv.Itoa(128))
	values.Set("lr", strconv.FormatFloat(0.01, 'f', -1, 32))


	final := routerUrl + "/" + functionName + "?" + values.Encode()
	fmt.Println(final)

	resp, err := http.Get(final)
	panicIf(err)

	var exit []string
	body, err := ioutil.ReadAll(resp.Body)
	err = json.Unmarshal(body, &exit)
	panicIf(err)
	fmt.Println(string(body), exit)



}