package model

import (
	"github.com/RedisAI/redisai-go/redisai"
	"gorgonia.org/tensor"
	"sync"

	"log"
)

const (
	weightSuffix   = "-weights"
	biasSuffix     = "-bias"
	gradientSuffix = "-grad"
)

type (

	// Holds the Layers of the model
	Model struct {
		Name       string
		LayerNames []string
		Layers     []*Layer

		Lr float64

		redisClient redisai.Client

		// Internal Lock to be applied during the update
		mu sync.Mutex
	}

	// Layer keeps the Weights and Bias of a certain layer of the Neural Network
	Layer struct {
		Name string

		WeightShape []int
		Weights     *tensor.Dense

		BiasShape int
		Bias      *tensor.Dense
	}

	// Gradient saves the gradients of a layer
	Gradient struct {
		WeightShape []int
		Weights     *tensor.Dense

		BiasShape int
		Bias      *tensor.Dense
	}
)

// Creates a new model with the specified layers
func NewModel(name string, layerNames []string, lr float64, client redisai.Client) *Model {
	return &Model{
		Name:       name,
		LayerNames: layerNames,
		Lr: lr,
		redisClient: client,
	}
}

// Build gets all the initialized layers from the database
// Build should be called once just after the network is initialized by a worker
func (m *Model) Build(psId string)  {
	// For each layer name create a new layer with the tensors from the database
	for _, layerName := range m.LayerNames {
		l, _ := NewLayer(m.redisClient, layerName, psId)
		m.Layers = append(m.Layers, l)
	}
}

// Update applies a set of gradients to all the layers
// Simply iterate through the model layers and update each with the gradients
// Simply use the layer names of the model with the -bias-grad added to them
// TODO this should take as a parameter an Id so we can get the gradients of a specific worker
func (m *Model) Update(psId, funcId string) error {

	// lock the model
	m.mu.Lock()
	defer m.mu.Unlock()

	for Idx, layerName := range m.LayerNames {

		// Get the gradients from the database
		g, err := NewGradient(m.redisClient, layerName, psId, funcId)
		if err != nil {
			log.Fatal("Could not build gradient", err)
			return err
		}

		// Update the layer
		err = m.Layers[Idx].Update(g, m.Lr)
		if err != nil {
			log.Fatal("Could not update layer", layerName, err)
			return err
		}

	}

	return nil
}

// Build a new layer by getting it from the database already initialized
func NewLayer(redisClient redisai.Client, name, psId string) (*Layer, error) {

	weightName, biasName := getWeightKeys(name, false, psId, "")

	// Get the weight and bias array from the redis database
	_, sWeights, weightValues, err := redisClient.TensorGetValues(weightName)
	_, sBias, biasValues, err := redisClient.TensorGetValues(biasName)

	if err != nil {
		log.Fatal("Unable to retrieve the values from the database", err)
		return nil, err
	}

	// Cast the shape to an int array and build the layer tensor
	dimWeights := shapeToIntArray(sWeights...)
	dimBias := shapeToIntArray(sBias...)

	// Build the actual tensors
	w := tensor.New(tensor.WithShape(dimWeights...), tensor.WithBacking(weightValues))
	b := tensor.New(tensor.WithShape(dimBias...), tensor.WithBacking(biasValues))

	return &Layer{
		Name:        name,
		WeightShape: dimWeights,
		Weights:     w,
		BiasShape:   dimBias[0],
		Bias:        b,
	}, nil

}

// Update the layer given a particular gradient using SGD and the given learning rate
func (layer *Layer) Update(g *Gradient, lr float64) error {

	// Update the gradients with the learning rate
	err := g.applyLR(lr)
	if err != nil {
		log.Fatal("Error when applying gradients", err)
		return err
	}

	// Subtract the gradients from the layer
	layer.Weights, _ = layer.Weights.Sub(g.Weights)
	layer.Bias, _ = layer.Bias.Sub(g.Bias)

	return nil
}

// Reads a gradient from the database
func NewGradient(redisClient redisai.Client, layerName , psId, funcId string) (*Gradient, error) {

	// Get the redis keys
	weightName, biasName := getWeightKeys(layerName, true, psId, funcId)

	// Get the weight and bias array from the redis database
	_, sWeights, weightValues, err := redisClient.TensorGetValues(weightName)
	_, sBias, biasValues, err := redisClient.TensorGetValues(biasName)

	if err != nil {
		log.Fatal("Unable to retrieve the values from the database", err)
		return nil, err
	}

	// Cast the shape to an int array and build the layer tensor
	dimWeights := shapeToIntArray(sWeights...)
	dimBias := shapeToIntArray(sBias...)

	// Build the actual tensors
	w := tensor.New(tensor.WithShape(dimWeights...), tensor.WithBacking(weightValues))
	b := tensor.New(tensor.WithShape(dimBias...), tensor.WithBacking(biasValues))

	return &Gradient{
		WeightShape: dimWeights,
		Weights:     w,
		BiasShape:   dimBias[0],
		Bias:        b,
	}, nil

}

// Multiplies the weights and bias by the learning rate before applying it to a Layer in an update
func (g *Gradient) applyLR(lr float64) error {

	var err error
	g.Weights, err = g.Weights.MulScalar(lr, false)
	g.Bias, err = g.Bias.MulScalar(lr, false)

	if err != nil {
		log.Fatal("Error when multiplying the gradients by the LR", err)
		return err
	}

	return nil
}
