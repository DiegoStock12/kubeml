package model

import (
	"github.com/hashicorp/go-multierror"
	"go.uber.org/zap"
)

// Abstraction so we can implement SGD
type (
	Optimizer interface {
		Step(m *Model, funcId string) error
	}


	SGD struct {
		lr float32
	}
)

// Applies the learning rate to the gradient
func (sgd SGD) applyLR(g *Gradient) error {
	var finalErr *multierror.Error
	var err error

	g.Weights, err = g.Weights.MulScalar(sgd.lr, false)
	finalErr = multierror.Append(finalErr, err)

	g.Bias, err = g.Bias.MulScalar(sgd.lr, false)
	finalErr = multierror.Append(finalErr, err)

	return finalErr.ErrorOrNil()

}

// Updates the layer with the given gradient
func (sgd SGD) updateLayer(g *Gradient, l *Layer)  error {
	// update the gradients with the learning rate
	err := sgd.applyLR(g)
	if err != nil {
		return err
	}

	// Subtract the gradients from the layer
	l.Weights, _ = l.Weights.Sub(g.Weights)
	l.Bias, _ = l.Bias.Sub(g.Bias)

	return nil
}

// TODO this should replace the model update stuff
// Step updates all the parameters in the network
func (sgd SGD) Step(m *Model, funcId string)  error {
	// update each of the Layers
	// lock the model
	m.mu.Lock()
	defer m.mu.Unlock()

	for idx, layerName := range m.LayerNames {

		// Get the gradients from the database
		g, err := newGradient(m.redisClient, layerName, m.psId, funcId)
		if err != nil {
			m.logger.Error("Could not build gradient",
				zap.String("layer", layerName),
				zap.Error(err))
			return err
		}

		// TODO the layer should not have the responsibility of choosing the algorithm
		// update the layer
		err = m.Layers[idx].update(g, m.lr)
		if err != nil {
			m.logger.Error("Could not update layer",
				zap.String("layer",layerName),
				zap.Error(err))
			return err
		}

	}

	return nil
}

