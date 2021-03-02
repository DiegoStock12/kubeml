package main

import "fmt"

func pIf(err error) {
	if err != nil {
		panic(err)
	}
}


type Example struct {
	Par int `json:"par"`
}

func main() {
	 ch := make(chan bool, 5)

	for i := 0; i < 5; i++ {
		ch <- true
	}

	close(ch)

	fmt.Println("Iterating once")
	for j := range ch {
		fmt.Println(j)
	}

	fmt.Println("Iterating again")
	for j := range ch {
		fmt.Println(j)
	}

}
