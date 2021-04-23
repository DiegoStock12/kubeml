package util

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/gomodule/redigo/redis"
	"time"
)

// number of commands before a pipeline flush
const pipelinePeriod = 50

var redisUrl = fmt.Sprintf("redis://%v:%v", api.RedisUrl, api.RedisPort)

// GetRedisConnectionPool creates and returns a redis connection pool
// which will be used when asking for a redisai connection in the future
func GetRedisConnectionPool() *redis.Pool {
	return &redis.Pool{
		Dial: func() (redis.Conn, error) {
			return redis.DialURL(redisUrl)
		},
		MaxIdle:     5,
		IdleTimeout: 240 * time.Second,
	}
}

// GetRedisAIClient returns a connection from the previously created pool of the
// trainjob. It optionally activates pipelining upon request
func GetRedisAIClient(pool *redis.Pool, pipeline bool) *redisai.Client {
	client := redisai.Connect("", pool)

	if pipeline {
		client.Pipeline(pipelinePeriod)
	}

	return client

}
