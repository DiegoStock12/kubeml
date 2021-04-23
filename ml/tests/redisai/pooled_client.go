package main

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/gomodule/redigo/redis"
	"gorgonia.org/tensor"
	"time"
)

var redisDockerAddress = "192.168.99.102"

var redisUrl = fmt.Sprintf("redis://%v:%v", redisDockerAddress, api.RedisPort)


func tryClient(pool *redis.Pool, id int) {

	shape := []int{3, 4}
	// create a tensor and setting
	t := tensor.Ones(tensor.Float32, shape...)
	fmt.Println(t)
	t, err := t.MulScalar(float32(id), false)
	if err != nil {
		panic(err)
	}

	// get the client
	client := redisai.Connect("", pool)
	defer client.Close()

	for i := 0; i < 20; i++ {
		name := fmt.Sprintf("worker-%v-tensor%v", id, i)
		err := client.TensorSet(name, redisai.TypeFloat32, []int64{3, 4}, t.Data())
		if err != nil {panic(err)}
	}

	// read them
	client.Pipeline(20)
	for i := 0; i < 20; i++ {
		name := fmt.Sprintf("worker-%v-tensor%v", id, i)
		_, _, _, err := client.TensorGetValues(name)
		if err != nil {panic(err)}
	}

	client.Flush()

	for i := 0; i < 20; i++ {
		resp, err := client.Receive()

		err, _, _, data := redisai.ProcessTensorGetReply(resp, err)
		if err != nil {
			panic(err)
		}
		fmt.Println("worker", id, data)

	}

	fmt.Println("worker", id, "done")
}

func main() {

	// define the pool
	pool := &redis.Pool{
		Dial: func() (redis.Conn, error) {
			return redis.DialURL(redisUrl)
		},
		MaxIdle:   5,
		IdleTimeout: 1 * time.Second,
	}

	// create the redis client based on this pool

	for i := 0; i < 5; i++ {
		fmt.Println("starting worker", i)
		go tryClient(pool, i)
		
	}

	fmt.Println("sleeping...")
	time.Sleep(2 * time.Second)

	for i := 0; i < 5; i++ {
		fmt.Println(pool.ActiveCount(), pool.Stats(), pool.IdleCount())
		time.Sleep(1 * time.Second)
	}

}
