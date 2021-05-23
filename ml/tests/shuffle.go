package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"
)


func shuffle(layers []string, id int) {

	rand.Seed(int64(id))

	words := make([]string, len(layers))
	copy(words, layers)


	rand.Shuffle(len(words), func(i, j int) {
		words[i], words[j] = words[j], words[i]
	})

	fmt.Println("function is", id, "words is", words)

}

func main()  {


	layers := strings.Fields("weights1 weights2 weights3 weights4")

	for i := 0; i < 4; i++ {
		go shuffle(layers, i)
	}

	time.Sleep(1 * time.Second)

}
