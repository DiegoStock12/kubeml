package main

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/gomodule/redigo/redis"
	"log"
	"reflect"
)



func main() {

	// Connect to the client
	conn := redisai.Connect(fmt.Sprintf("redis://%s:%d", "192.168.99.101", 31618), nil)
	defer conn.Close()

	// Get if the key exists
	res, err := redis.Int(conn.DoOrSend("EXISTS", redis.Args{"example:conv1-weight-grad/0"}, nil))
	if err != nil {
		log.Fatal(err)
	}



	//for _, layer := range strings.Split(l, " ") {
	//	fmt.Println(layer)
	//}
	fmt.Println(reflect.TypeOf(res), res)


}
