package model

import (
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/pkg/errors"
	"go.uber.org/zap"
)

type (

	// ParallelSGD simply averages the weights of the models
	// trained independently. In this way, we get the freedom of
	// using any optimizer in the functions.
	//
	// Simply fetch all the model weights and average them
	ParallelSGD struct {
		logger *zap.Logger
	}
)

func MakeParallelSGD(logger *zap.Logger) ParallelSGD {
	return ParallelSGD{logger: logger.Named("parallel-sgd")}
}

// Average averages the layers by the number of finished functions
func (psgd ParallelSGD) Average(m *Model, num int) error {

	psgd.logger.Debug("Averaging", zap.Int("num", num))

	var err error
	for _, entry := range m.StateDict {
		// divide the sum of the layer weights by the
		switch entry.layer.Dtype {
		case redisai.TypeFloat32:
			entry.layer.Weights, err = entry.layer.Weights.DivScalar(float32(num), true)
			if err != nil {
				psgd.logger.Error("Error dividing weights",
					zap.Error(err))
				return errors.Wrap(err, "error dividing float weights")
			}

		case redisai.TypeInt64:
			entry.layer.Weights, err = entry.layer.Weights.DivScalar(int64(num), true)
			if err != nil {
				psgd.logger.Error("Error dividing weights",
					zap.Error(err))
				return errors.Wrap(err, "error diving int weights")
			}
		}
	}

	return nil

}
