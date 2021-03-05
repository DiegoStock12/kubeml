package main

import (
	"fmt"
	"sync"
	"time"
)

func doStuff(wg *sync.WaitGroup, id int) {
	fmt.Println("about to wait", id)
	wg.Wait()

	fmt.Println("accessing...", id)
	wg.Add(1)
	go func() {
		defer wg.Done()
		time.Sleep(5 * time.Second)
		fmt.Println("finished", id)
	}()
}

func main() {

	wg := &sync.WaitGroup{}

	go doStuff(wg, 1)
	time.Sleep(1 * time.Second)

	go doStuff(wg, 2)
	go doStuff(wg, 3)
	go doStuff(wg, 4)

	time.Sleep(15 * time.Second)

}
