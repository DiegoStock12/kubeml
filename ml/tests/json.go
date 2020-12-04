package main

import (
	"encoding/json"
	"errors"
	"fmt"
)

func pIf(err error) {
	if err != nil {
		panic(err)
	}
}


type Example struct {
	Par int `json:"par"`
}

func main() {
	err := errors.New("Sample error")

	e := Example{
		Par: 2,
	}

	j, err  := json.Marshal(e)
	fmt.Println("marshalled json")
	pIf(err)


	var test Example
	err = json.Unmarshal(j,&test)
	pIf(err)

	fmt.Printf("%v",test)

}
