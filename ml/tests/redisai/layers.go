package main

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/gomodule/redigo/redis"
	"log"
)



func main() {

	// Connect to the client
	conn := redisai.Connect(fmt.Sprintf("redis://%s:%d", "192.168.99.102", 6379), nil)
	defer conn.Close()

	// Get the values of the list
	l, err := redis.Strings(conn.DoOrSend("LRANGE", redis.Args{"layers", 0, -1}, nil))
	if err != nil {
		log.Fatal(err)
	}


	//for _, layer := range strings.Split(l, " ") {
	//	fmt.Println(layer)
	//}
	fmt.Println(l)


}
