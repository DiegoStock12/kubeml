package ps

import (
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/thesis/ml/pkg/model"
)

type (
	ParameterServer struct {
		psId string

		model *model.Model

		redisClient redisai.Client

	}
)
