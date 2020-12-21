package model

import "go.uber.org/zap"

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
	return ParallelSGD{logger: logger}
}

// Merge fetches weights from the database and averages them to create a new
// reference model for the training job
// TODO we can parallelize fetching and averaging each layer
func (psgd ParallelSGD) Merge(m *Model, funcs ...int)  {
	psgd.logger.Debug("Merging layers...", zap.Any("funcs", funcs))


	sd := make(map[string]*Layer)

	// For each of the layers get and average the weights
	// from all the functions
	for layerName := range m.StateDict {
		psgd.logger.Debug("Merging layer", zap.String("name", layerName))
		num := 0

		for _, fId := range funcs {
			layer, err := newLayer(psgd.logger, m.redisClient, layerName, m.jobId, fId)
			if err != nil {
				psgd.logger.Error("Could not load layer from database",
					zap.Error(err),
					zap.String("name", layerName),
					zap.Int("funcId", fId))
				continue
			}

			// if the layer is not set in the statedict set it
			// with the current layer. If it is there simply add the
			// newly fetched layer
			total, exists := sd[layerName]
			if !exists {
				sd[layerName] = layer
			} else {
				total.Weights, err  = total.Weights.Add(layer.Weights)
				if err != nil {
					psgd.logger.Error("Error adding weights",
						zap.Error(err))
				}
				total.Bias, err = total.Bias.Add(layer.Bias)
				if err != nil {
					psgd.logger.Error("Error adding bias",
						zap.Error(err))
				}
			}

			num++

		}

		psgd.logger.Debug("Nums are", zap.Int("num", num))
		// Finally average all the weights and biases and set the num to 0
		layer := sd[layerName]
		var err error
		layer.Weights, err = layer.Weights.DivScalar(float32(num), true)
		if err != nil {
			psgd.logger.Error("Error dividing weights",
				zap.Error(err))
		}
		layer.Bias, err = layer.Bias.DivScalar(float32(num), true)
		if err != nil {
			psgd.logger.Error("Error dividing bias",
				zap.Error(err))
		}

	}

	// in the end simply apply the statedict to the model
	m.StateDict = sd
}

