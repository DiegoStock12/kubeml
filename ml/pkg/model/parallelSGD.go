package model

import (
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

// Merge fetches weights from the database and averages them to create a new
// reference model for the training job
func (psgd ParallelSGD) Merge(m *Model, funcs ...int) {
	psgd.logger.Debug("Merging layers...", zap.Any("funcs", funcs))

	// Fetch the layers output by each of the training functions
	// For each of the layers accumulate them and then in the end divide
	// by the total number of successful functions
	// Then update the state dict
	sd := make(map[string]*Layer)
	for layerName := range m.StateDict {
		psgd.logger.Debug("Merging layer", zap.String("name", layerName))
		num := 0

		// First fetch the layer output by all functions
		// if it doesn't exists in the new state dict assign it,
		// if it does, simply add it
		for _, fId := range funcs {
			layer, err := m.NewLayer(layerName, fId)
			if err != nil {
				psgd.logger.Error("Could not load layer from database",
					zap.Error(err),
					zap.String("name", layerName),
					zap.Int("funcId", fId))
				continue
			}

			if total, exists := sd[layerName]; !exists {
				sd[layerName] = layer
			} else {
				total.Weights, err = total.Weights.Add(layer.Weights)
				if err != nil {
					psgd.logger.Error("Error adding weights",
						zap.Error(err))
				}
				if layer.HasBias {
					total.Bias, err = total.Bias.Add(layer.Bias)
					if err != nil {
						psgd.logger.Error("Error adding bias",
							zap.Error(err))
					}
				}
			}
			num++

		}

		// Finally average all the weights and biases and set the num to 0
		layer := sd[layerName]
		var err error
		layer.Weights, err = layer.Weights.DivScalar(float32(num), true)
		if err != nil {
			psgd.logger.Error("Error dividing weights",
				zap.Error(err))
		}

		if layer.HasBias {
			layer.Bias, err = layer.Bias.DivScalar(float32(num), true)
			if err != nil {
				psgd.logger.Error("Error dividing bias",
					zap.Error(err))
			}

		}

	}

	// in the end simply apply the statedict to the model
	m.StateDict = sd
}
