package model

import (
	"github.com/hashicorp/go-multierror"
	"go.uber.org/zap"
)

// Abstraction so we can implement SGD
type (
	Optimizer interface {
		Step(m *Model, funcId int, N int) error
	}

	// SGD represents the Stochastic Gradient Descent
	// optimization method
	SGD struct {
		Lr float32
	}

	// Adam is the Adam optimizer
	Adam struct {
		Lr float32
		Betas []float32
		Eps float32
	}
)

// Applies the learning rate to the gradient
// we divide the lr by the number of functions
func (sgd SGD) applyLR(g *Gradient, N int) error {
	var finalErr *multierror.Error
	var err error

	g.Weights, err = g.Weights.MulScalar(sgd.Lr/float32(N), false)
	finalErr = multierror.Append(finalErr, err)

	g.Bias, err = g.Bias.MulScalar(sgd.Lr/float32(N), false)
	finalErr = multierror.Append(finalErr, err)

	return finalErr.ErrorOrNil()

}

// Updates the layer with the given gradient
func (sgd SGD) updateLayer(g *Gradient, l *Layer, N int)  error {
	// update the gradients with the learning rate
	err := sgd.applyLR(g, N)
	if err != nil {
		return err
	}

	// Subtract the gradients from the layer
	l.Weights, _ = l.Weights.Sub(g.Weights)
	l.Bias, _ = l.Bias.Sub(g.Bias)

	return nil
}

// Step updates all the parameters in the network
// given the function id that we need to fetch from the
// database, and the N, so we can give and approximate importante
// to each of the updates
func (sgd SGD) Step(m *Model, funcId int, N int)  error {
	// update each of the Layers
	// lock the model
	m.mu.Lock()
	defer m.mu.Unlock()

	for name, layer := range m.StateDict {

		// Get the gradients from the database
		grad, err := newGradient(m.redisClient, name, m.jobId, funcId)
		if err != nil {
			m.logger.Error("Could not build gradient",
				zap.String("layer", name),
				zap.Error(err))
			return err
		}

		// update the layer
		err = sgd.updateLayer(grad, layer, N)
		if err != nil {
			m.logger.Error("Could not update layer",
				zap.String("layer",name),
				zap.Error(err))
			return err
		}

	}

	return nil
}

