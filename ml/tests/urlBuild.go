package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"time"
)


const (
	routerUrl = "http://192.168.99.101:32422"
	functionName = "knetwork"
)

func panicIf(err error) {
	if err != nil {
		panic(err)
	}
}


func main() {

	start := time.Now()
	values := url.Values{}
	values.Set("task", "init")
	values.Set("psId", "test")
	values.Set("N", "1")
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

	fmt.Println(time.Now().Sub(start))



}