package model

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/gomodule/redigo/redis"
	"go.uber.org/zap"
	"gorgonia.org/tensor"
	"sync"
)

type (

	// Holds the Layers of the model
	Model struct {
		logger *zap.Logger

		// Id of the parameter server
		psId string

		Name       string
		LayerNames []string
		Layers     []*Layer

		// lr must be float32 to be the same type as the tensors
		lr      float32
		lrSched LrScheduler

		redisClient *redisai.Client

		// Internal Lock to be applied during the update
		// TODO looks like each tensor has its own lock. If this is the case maybe we can speed things up
		mu sync.Mutex
	}

	// Layer keeps the Weights and Bias of a certain layer of the Neural Network
	Layer struct {
		Name string

		WeightShape []int64
		Weights     *tensor.Dense

		HasBias   bool
		BiasShape []int64
		Bias      *tensor.Dense
	}

	// Gradient saves the gradients of a layer
	Gradient struct {
		WeightShape []int64
		Weights     *tensor.Dense

		HasBias   bool
		BiasShape []int64
		Bias      *tensor.Dense
	}

	// Just a learning rate scheduler that multiplies the rate by rate when invoked
	LrScheduler struct {
		rate float32
	}
)

// Creates a new model with the specified layers
func NewModel(logger *zap.Logger, psId, name string, layerNames []string, lr float32, client *redisai.Client) *Model {
	return &Model{
		logger:      logger.Named("model"),
		Name:        name,
		psId:        psId,
		LayerNames:  layerNames,
		lr:          lr,
		redisClient: client,
	}
}

// Build gets all the initialized layers from the database
// Build should be called once just after the network is initialized by a worker
func (m *Model) Build() error {
	// For each layer name create a new layer with the tensors from the database
	m.logger.Debug("Building the model", zap.String("psId", m.psId))

	for _, layerName := range m.LayerNames {

		m.logger.Debug("Creating new layer", zap.String("layerName", layerName))
		l, err := newLayer(m.logger, m.redisClient, layerName, m.psId)
		if err != nil {
			m.logger.Error("Error building layer",
				zap.String("layer", layerName),
				zap.Error(err))
			return err
		}
		m.Layers = append(m.Layers, l)
	}

	return nil
}

// Update applies a set of gradients to all the layers
// Simply iterate through the model layers and update each with the gradients
// Simply use the layer names of the model with the -bias-grad added to them
// TODO seems like the layers already have a lock so maybe we do not need the mutex here
func (m *Model) Update(funcId string) error {

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

		// update the layer
		err = m.Layers[idx].update(g, m.lr)
		if err != nil {
			m.logger.Error("Could not update layer",
				zap.String("layer", layerName),
				zap.Error(err))
			return err
		}

	}

	return nil
}

// Summary runs through the layers of a model and prints its info
func (m *Model) Summary() {
	for i, n := range m.LayerNames {
		m.logger.Info("Layer",
			zap.String("name", n),
			zap.Any("shape", m.Layers[i].WeightShape),
			zap.Bool("bias", m.Layers[i].HasBias),
		)
	}

}

// Save saves the new updated weights and bias in the database so it can be retrieved
// by the following functions
// TODO we could use pipeline to speed it up
func (m *Model) Save() error {
	m.logger.Info("Publishing model on the database")

	for i, layerName := range m.LayerNames {

		m.logger.Debug("Setting weights", zap.String("layer", layerName), zap.Any("shape", m.Layers[i].Weights))
		args, _ := makeArgs(layerName, m.Layers[i].WeightShape, m.Layers[i].Weights.Data())
		_, err := m.redisClient.DoOrSend("AI.TENSORSET", *args, nil)
		if err != nil {
			m.logger.Error("Error setting weights",
				zap.String("layer", layerName),
				zap.Error(err))
			return err
		}

		// Set the bias only if it is needed
		if m.Layers[i].HasBias {
			m.logger.Debug("Setting bias", zap.String("layer", layerName), zap.Any("shape", m.Layers[i].Bias))
			args, _ = makeArgs(layerName, m.Layers[i].BiasShape, m.Layers[i].Bias.Data())
			_, err = m.redisClient.DoOrSend("AI.TENSORSET", *args, nil)
			if err != nil {
				m.logger.Error("Error setting bias",
					zap.String("layer", layerName),
					zap.Error(err))
				return err
			}
		}

	}

	m.logger.Info("Model published in the DB")
	return nil

}

// Build a new layer by getting it from the database already initialized
func newLayer(logger *zap.Logger, redisClient *redisai.Client, name, psId string) (*Layer, error) {

	// Get the redis keys
	weightName, biasName := getWeightKeys(name, false, psId, "")

	// Build the weight tensor
	logger.Debug("Loading the weights...")
	_, sWeights, weightValues, err := redisClient.TensorGetValues(weightName)
	if err != nil {
		return nil, err
	}
	dimWeights := shapeToIntArray(sWeights...)
	w := tensor.New(tensor.WithShape(dimWeights...), tensor.WithBacking(weightValues))

	// If we have to build the bias tensor
	var b *tensor.Dense
	var sBias []int64
	biasExists, err := tensorExists(redisClient, biasName)
	if err != nil {
		return nil, err
	}

	hasBias := true
	if biasExists {
		logger.Debug("Loading the biases")
		_, sBias, biasValues, err := redisClient.TensorGetValues(biasName)
		if err != nil {
			return nil, err
		}
		// Cast the shape to an int array and build the layer tensor
		dimBias := shapeToIntArray(sBias...)
		// Build the actual tensor
		b = tensor.New(tensor.WithShape(dimBias...), tensor.WithBacking(biasValues))
		hasBias = true
	}

	return &Layer{
		Name:        name,
		WeightShape: sWeights,
		Weights:     w,
		HasBias:     hasBias,
		BiasShape:   sBias,
		Bias:        b,
	}, nil

}

// update the layer given a particular gradient using SGD and the given learning rate
func (layer *Layer) update(g *Gradient, lr float32) error {

	// update the gradients with the learning rate
	err := g.applyLR(lr)
	if err != nil {
		return err
	}

	// Subtract the gradients from the layer
	layer.Weights, _ = layer.Weights.Sub(g.Weights)

	// Just update if the bias is set
	if layer.HasBias {
		layer.Bias, _ = layer.Bias.Sub(g.Bias)
	}

	return nil
}

// Reads a gradient from the database
func newGradient(redisClient *redisai.Client, layerName, psId, funcId string) (*Gradient, error) {

	// Get the redis keys
	weightName, biasName := getWeightKeys(layerName, true, psId, funcId)

	// Build the weight tensor
	_, sWeights, weightValues, err := redisClient.TensorGetValues(weightName)
	if err != nil {
		return nil, err
	}
	dimWeights := shapeToIntArray(sWeights...)
	w := tensor.New(tensor.WithShape(dimWeights...), tensor.WithBacking(weightValues))

	// If we have to build the bias tensor
	var b *tensor.Dense
	var sBias []int64
	biasExists, err := tensorExists(redisClient, biasName)
	if err != nil {
		return nil, err
	}

	hasBias := false
	if biasExists {
		_, sBias, biasValues, err := redisClient.TensorGetValues(biasName)
		if err != nil {
			return nil, err
		}
		// Cast the shape to an int array and build the layer tensor
		dimBias := shapeToIntArray(sBias...)
		// Build the actual tensor
		b = tensor.New(tensor.WithShape(dimBias...), tensor.WithBacking(biasValues))
		hasBias = true
	}

	return &Gradient{
		WeightShape: sWeights,
		Weights:     w,
		HasBias: hasBias,
		BiasShape:   sBias,
		Bias:        b,
	}, nil

}

// Multiplies the weights and bias by the learning rate before applying it to a Layer in an update
func (g *Gradient) applyLR(lr float32) error {

	var err error
	g.Weights, err = g.Weights.MulScalar(lr, false)
	g.Bias, err = g.Bias.MulScalar(lr, false)

	if err != nil {
		return err
	}

	return nil
}

// Sets the model learning rate to the new value
func (lrs LrScheduler) updateLr(m *Model) {
	m.logger.Info("Updating the LR",
		zap.Float32("Rate", lrs.rate),
		zap.Float32("Current rate", m.lr))
	m.lr *= lrs.rate
}

// tensorExists simply returns whether the tensor is present in the cache
// In some networks (i.e resnets) the bias of the layers is not used, so in those
// cases it will not be published. In this case we can see whether that is true
func tensorExists(client *redisai.Client, tensorName string) (bool, error) {
	res, err := redis.Int(client.DoOrSend("EXISTS", redis.Args{tensorName}, nil))
	if err != nil {
		return false, err
	}

	// we get a 1 if it exists and a 0 if it doesn't
	switch res {
	case 0:
		return false, err
	case 1:
		return true, nil
	default:
		return false, fmt.Errorf("received unknown result from the cache: %v", res)
	}

}
