package main

import (
	"encoding/json"
	"fmt"
)

type Test struct {
	Name string `json:"name"`
	Channel chan bool `json:"-"`
}

func main(){

	t := Test{
		Name:    "Diego",
		Channel: make(chan bool),
	}

	fmt.Println("created test object", t)

	body, err := json.Marshal(t)
	if err != nil {panic(err)}

	fmt.Println(string(body))


}
