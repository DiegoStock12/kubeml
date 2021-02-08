package main

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/gomodule/redigo/redis"
	"reflect"
)

const psID = "026c8b93"

func main() {

	client := redisai.Connect(fmt.Sprintf("redis://%s:%d", api.REDIS_ADDRESS_DEBUG, api.REDIS_PORT_DEBUG), nil)

	//filter := fmt.Sprintf("%s*/*", psID)
	// list the tensor names
	filter := "*"
	args := redis.Args{filter}
	reply, err := redis.Strings(client.DoOrSend("KEYS", args, nil))
	if err != nil{panic(err)}

	fmt.Println(reply)
	deleteArgs := redis.Args{}.AddFlat(reply)
	deleteArgs = deleteArgs.AddFlat(reply)
	fmt.Println(reflect.TypeOf(reply), reply, len(reply))
	for _, k := range reply{
		fmt.Println(k)
	}

	fmt.Println(deleteArgs)
	//r, err := client.DoOrSend("DEL", deleteArgs, nil)
	//if err != nil {
	//	panic(err)
	//}
	//fmt.Println(r)
}
