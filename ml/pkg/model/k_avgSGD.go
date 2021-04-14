package model

import (
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/pkg/errors"
	"go.uber.org/zap"
)

type (

	// KavgSGD simply averages the weights of the models
	// trained independently. In this way, we get the freedom of
	// using any optimizer in the functions.
	//
	// Simply fetch all the model weights and average them
	KavgSGD struct {
		logger *zap.Logger
		model  *Model
	}
)

func MakeKavgSGD(logger *zap.Logger) KavgSGD {
	return KavgSGD{logger: logger.Named("parallel-sgd")}
}

// Step averages the models returned by the functions into one single
// model, which will be used in the next epoch by all functions
func (avg KavgSGD) Step(funcs ...int) error {

	num := len(funcs)
	avg.logger.Debug("Averaging", zap.Int("num", num))

	var err error
	for _, layer := range avg.model.StateDict {
		// divide the sum of the layer weights by the
		switch layer.Dtype {
		case redisai.TypeFloat32:
			layer.Weights, err = layer.Weights.DivScalar(float32(num), true)
			if err != nil {
				avg.logger.Error("Error dividing weights",
					zap.Error(err))
				return errors.Wrap(err, "error dividing float weights")
			}

		case redisai.TypeInt64:
			layer.Weights, err = layer.Weights.DivScalar(int64(num), true)
			if err != nil {
				avg.logger.Error("Error dividing weights",
					zap.Error(err))
				return errors.Wrap(err, "error diving int weights")
			}
		}
	}

	return nil

}

// Save publishes the model to the database to be loaded by the functions
// in the following epoch
// K-avg merges the models into a single reference model and publishes it
func (avg KavgSGD) Save(_ ...int) error {

	avg.logger.Info("Publishing model on the database")

	// start the transaction in the redis client
	avg.model.redisClient.DoOrSend("MULTI", nil, nil)
	for name, layer := range avg.model.StateDict {
		avg.model.logger.Debug("Setting layer", zap.String("name", name))
		err := avg.model.setLayer(name, layer)
		if err != nil {
			return err
		}
	}

	// execute all commands as a batch and empty response buffer
	_, err := avg.model.redisClient.ActiveConn.Do("EXEC")
	if err != nil {
		return errors.Wrap(err, "could not save tensors")
	}

	avg.model.logger.Info("Model published in the DB")
	return nil

}

// PreTrain performs the model wiping before the next iteration begins
func (avg KavgSGD) PreTrain() error {
	avg.model.Clear()
	return nil
}

func (avg KavgSGD) Update(funcId int) error {

	avg.logger.Debug("Updating model layers",
		zap.Int("funcId", funcId))

	// lock the model, only one thread can use the
	// redis client concurrently
	avg.model.mu.Lock()
	defer avg.model.mu.Unlock()

	// load the function layers
	for _, layer := range avg.model.layerNames {
		err := avg.model.fetchLayer(layer, funcId)
		if err != nil {
			avg.logger.Error("could not fetch layer",
				zap.Error(err),
				zap.String("name", layer),
				zap.Int("funcId", funcId))
			return err
		}
	}

	avg.model.redisClient.Flush()

	for _, layerName := range avg.model.layerNames {
		layer, err := avg.model.buildLayer(layerName)
		if err != nil {
			avg.logger.Error("Could not build layer from database",
				zap.Error(err),
				zap.String("name", layerName),
				zap.Int("funcId", funcId))
			return err
		}

		if total, exists := avg.model.StateDict[layerName]; !exists {
			avg.model.StateDict[layerName] = layer
		} else {
			total.Weights, err = total.Weights.Add(layer.Weights)
			if err != nil {
				avg.logger.Error("Error adding weights",
					zap.Error(err))

				return err
			}
		}
	}

	avg.logger.Debug("Model updated",
		zap.Int("funcId", funcId))

	return nil

}
