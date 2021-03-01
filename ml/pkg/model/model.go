package model

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/gomodule/redigo/redis"
	"github.com/pkg/errors"
	"go.uber.org/zap"
	"gorgonia.org/tensor"
	"sync"
)

const (
	// Constants to save and retrieve the gradients
	WeightSuffix = ".weight"
	BiasSuffix   = ".bias"
)

type (

	// Holds the Layers of the model
	Model struct {
		logger *zap.Logger

		// Id of the parameter server
		jobId string

		Name string

		// StateDict holds the layer names
		// and the layers of the model. Each
		// layer has a bias and a weight
		StateDict map[string]*Layer

		// layerNames holds the names of the layers
		// which will be used to build the model for the
		// first time
		layerNames []string

		redisClient *redisai.Client

		// Internal Lock to be applied during the update
		// TODO looks like each tensor has its own lock. If this is the case maybe we can speed things up
		mu sync.Mutex
	}

	// Layer keeps the Weights and Bias of a certain layer of the Neural Network
	Layer struct {
		Name    string
		Weights *tensor.Dense
		HasBias bool
		Bias    *tensor.Dense
	}
)

// Creates a new model with the specified layers
func NewModel(
	logger *zap.Logger,
	jobId string,
	task api.TrainRequest,
	layerNames []string,
	client *redisai.Client) *Model {

	return &Model{
		logger:      logger.Named("model"),
		Name:        task.ModelType,
		jobId:       jobId,
		layerNames:  layerNames,
		StateDict:   make(map[string]*Layer),
		redisClient: client,
	}
}

// Build gets all the initialized layers from the database
// Build should be called once just after the network is initialized by a worker
func (m *Model) Build() error {
	// For each layer name create a new layer with the tensors from the database
	m.logger.Debug("Building the model", zap.String("jobId", m.jobId))

	for _, layerName := range m.layerNames {

		m.logger.Debug("Creating new layer", zap.String("layerName", layerName))
		l, err := m.NewLayer(layerName, -1)
		if err != nil {
			m.logger.Error("Error building layer",
				zap.String("layer", layerName),
				zap.Error(err))
			return err
		}

		// Add it to the statedict
		m.StateDict[layerName] = l
	}

	return nil
}

// Summary runs through the layers of a model and prints its info
func (m *Model) Summary() {
	for name, layer := range m.StateDict {
		m.logger.Info("Layer",
			zap.String("name", name),
			zap.Any("shape", layer.Weights.Shape()),
			zap.Bool("bias", layer.HasBias),
		)
	}

}

// Save saves the new updated weights and bias in the database so it can be retrieved
// by the following functions
func (m *Model) Save() error {
	m.logger.Info("Publishing model on the database")

	for name, layer := range m.StateDict {
		m.logger.Debug("Setting layer", zap.String("name", name))
		err := m.setLayer(name, layer)
		if err != nil {
			return err
		}
	}

	m.logger.Info("Model published in the DB")
	return nil

}

// SetLayer saves a layer's weights and bias if available in the storage
func (m *Model) setLayer(name string, layer *Layer) error {

	err := m.setWeights(name, layer)
	if err != nil {
		return err
	}

	if layer.HasBias {
		err = m.setBias(name, layer)
		if err != nil {
			return err
		}
	}

	return nil
}

func (m *Model) setWeights(name string, layer *Layer) error {
	args, _ := makeArgs(m.jobId, name, WeightSuffix, layer.Weights.Shape(), layer.Weights.Data())
	_, err := m.redisClient.DoOrSend("AI.TENSORSET", *args, nil)
	if err != nil {
		return errors.Wrapf(err, "could not set weights of layer %v", name)
	}
	return nil
}

func (m *Model) setBias(name string, layer *Layer) error {
	args, _ := makeArgs(m.jobId, name, BiasSuffix, layer.Bias.Shape(), layer.Bias.Data())
	_, err := m.redisClient.DoOrSend("AI.TENSORSET", *args, nil)
	if err != nil {
		return errors.Wrapf(err, "could not set bias of layer %v", name)
	}
	return nil

}

// NewLayer fetches a layer from the database. It first queries
// to see whether the layer contains bias or not, and then gets the layer
// indexed by the function ID which saved it in the first place.
//
// The function Id is used to build the tensor key name. If the funcID is
// >= 0, we know it is output from a function, if funcId is -1, we know
// that it is the reference model that we need to load and no suffix is added
// to the tensor name
func (m *Model) NewLayer(name string, funcId int) (*Layer, error) {

	// Get the redis keys
	weightName, biasName := getWeightKeys(name, m.jobId, funcId)
	sWeights, weightValues, err := fetchTensor(m.redisClient, weightName)
	if err != nil {
		return nil, err
	}

	dimWeights := shapeToIntArray(sWeights...)
	w := tensor.New(tensor.WithShape(dimWeights...), tensor.WithBacking(weightValues))

	// If we have to build the bias tensor
	var b *tensor.Dense
	biasExists, err := tensorExists(m.redisClient, biasName)
	if err != nil {
		return nil, err
	}

	hasBias := false
	if biasExists {
		sBias, biasValues, err := fetchTensor(m.redisClient, biasName)
		if err != nil {
			return nil, err
		}

		// Cast the shape to an int array and build the layer tensor
		dimBias := shapeToIntArray(sBias...)
		b = tensor.New(tensor.WithShape(dimBias...), tensor.WithBacking(biasValues))
		hasBias = true
	}

	return &Layer{
		Name:    name,
		Weights: w,
		HasBias: hasBias,
		Bias:    b,
	}, nil

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
