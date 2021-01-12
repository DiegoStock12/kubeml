package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"reflect"
)

func main() {

	var data []interface{}

	d, err := ioutil.ReadFile("./example.json")
	if err != nil {
		panic(err)
	}
	err = json.Unmarshal(d, &data)
	if err != nil {
		panic(err)
	}

	fmt.Println(data)
	fmt.Println(reflect.TypeOf(data[0]))



}
