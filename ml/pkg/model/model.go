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

	// Layer keeps the Weights of a certain layer of the Neural Network
	// the weights can be either the weights or bias indistinctly
	Layer struct {
		Name    string
		Weights *tensor.Dense
	}
)

// Creates a new model with the specified layers
func NewModel(
	logger *zap.Logger,
	jobId string,
	task api.TrainRequest,
	layerNames []string,
	client *redisai.Client) *Model {

	// set the client to use a pipeline
	client.Pipeline(10)

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

	// fetch the layers, they will be pipelined
	// so we perform one loop to query, flush, and then parse all in another loop
	for _, name := range m.layerNames {
		m.logger.Debug("Creating new layer", zap.String("layerName", name))

		err := m.fetchLayer(name, -1)
		if err != nil {
			m.logger.Error("Error building layer",
				zap.String("layer", name),
				zap.Error(err))
			return err
		}

	}

	err := m.redisClient.Flush()
	if err != nil {
		return errors.Wrap(err, "error flushing commands")
	}

	// parse all responses
	for _, name := range m.layerNames {
		layer, err := m.buildLayer(name)
		if err != nil {
			return errors.Wrapf(err, "error loading layer %s", name)
		}
		m.StateDict[name] = layer
	}

	return nil
}

// Summary runs through the layers of a model and prints its info
func (m *Model) Summary() {
	for name, layer := range m.StateDict {
		m.logger.Info("Layer",
			zap.String("name", name),
			zap.Any("shape", layer.Weights.Shape()),
		)
	}

}

// Save saves the new updated weights and bias in the database so it can be retrieved
// by the following functions
func (m *Model) Save() error {
	m.logger.Info("Publishing model on the database")

	// start the transaction in the redis client
	m.redisClient.DoOrSend("MULTI", nil, nil)
	for name, layer := range m.StateDict {
		m.logger.Debug("Setting layer", zap.String("name", name))
		err := m.setLayer(name, layer)
		if err != nil {
			return err
		}
	}

	// execute all commands as a batch and empty response buffer
	_, err := m.redisClient.ActiveConn.Do("EXEC")
	if err != nil {
		return errors.Wrap(err, "could not save tensors")
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

	return nil
}

func (m *Model) setWeights(name string, layer *Layer) error {
	args, _ := makeArgs(m.jobId, name, layer.Weights.Shape(), layer.Weights.Data())
	_, err := m.redisClient.DoOrSend("AI.TENSORSET", *args, nil)
	if err != nil {
		return errors.Wrapf(err, "could not set weights of layer %v", name)
	}
	return nil
}

//
//func (m *Model) setBias(name string, layer *Layer) error {
//	args, _ := makeArgs(m.jobId, name, BiasSuffix, layer.Bias.Shape(), layer.Bias.Data())
//	_, err := m.redisClient.DoOrSend("AI.TENSORSET", *args, nil)
//	if err != nil {
//		return errors.Wrapf(err, "could not set bias of layer %v", name)
//	}
//	return nil
//
//}

// fetchLayer calls the tensor get function in the pipelined client
// the results are pipelined and are thus gathered later on in the buildLayer
// function
func (m *Model) fetchLayer(name string, funcId int) error {

	// call get blob but ignore the results cause those are pipelined
	tensorName := getWeightKeys(name, m.jobId, funcId)
	_, _, _, err := m.redisClient.TensorGetBlob(tensorName)
	if err != nil {
		return err
	}

	return nil

}

// buildLayer reads the pipelined response, parses it and returns the Layer
func (m *Model) buildLayer(name string) (*Layer, error) {

	// get the next response in the pipelined client
	resp, err := m.redisClient.Receive()
	err, _, shapeInt64, blob := redisai.ProcessTensorGetReply(resp, err)
	if err != nil {
		return nil, err
	}

	values, err := blobToFloatArray(blob.([]byte), shapeInt64)
	if err != nil {
		return nil, err
	}
	shapeInt := shapeToIntArray(shapeInt64...)

	t := tensor.New(tensor.WithShape(shapeInt...), tensor.WithBacking(values))

	return &Layer{
		Name:    name,
		Weights: t,
	}, nil

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
	weightName := getWeightKeys(name, m.jobId, funcId)
	m.logger.Debug("Loading layer", zap.String("layer", weightName))
	sWeights, weightValues, err := fetchTensor(m.redisClient, weightName)
	if err != nil {
		return nil, err
	}

	dimWeights := shapeToIntArray(sWeights...)
	w := tensor.New(tensor.WithShape(dimWeights...), tensor.WithBacking(weightValues))

	return &Layer{
		Name:    name,
		Weights: w,
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
