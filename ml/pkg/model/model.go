package model

import (
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/diegostock12/kubeml/ml/pkg/util"
	"github.com/gomodule/redigo/redis"
	"github.com/pkg/errors"
	"go.uber.org/zap"
	"gorgonia.org/tensor"
	"math/rand"
	"sync"
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
		StateDict map[string]*Entry

		// layerNames holds the names of the layers
		// which will be used to build the model for the
		// first time
		layerNames []string

		redisPool *redis.Pool
	}

	// Layer keeps the Weights of a certain layer of the Neural Network
	// the weights can be either the weights or bias indistinctly
	Layer struct {
		Name    string
		Dtype   string
		Weights *tensor.Dense
	}

	// Entry acts like an envelope around layers to increase concurrency
	Entry struct {
		mu sync.Mutex
		layer *Layer
	}
)

// Creates a new model with the specified layers
func NewModel(
	logger *zap.Logger,
	jobId string,
	task api.TrainRequest,
	layerNames []string,
	pool *redis.Pool) *Model {

	return &Model{
		logger:     logger.Named("model"),
		Name:       task.ModelType,
		jobId:      jobId,
		layerNames: layerNames,
		StateDict:  make(map[string]*Entry),
		redisPool:  pool,
	}
}

// Build gets all the initialized layers from the database
// Build should be called once just after the network is initialized by a worker
func (m *Model) Build() error {
	// For each layer name create a new layer with the tensors from the database
	m.logger.Debug("Building the model", zap.String("jobId", m.jobId))

	// get the client
	redisClient := util.GetRedisAIClient(m.redisPool, true)
	defer redisClient.Close()

	// fetch the layers, they will be pipelined
	// so we perform one loop to query, flush, and then parse all in another loop
	for _, name := range m.layerNames {
		m.logger.Debug("Creating new layer", zap.String("layerName", name))

		err := m.fetchLayer(redisClient, name, -1)
		if err != nil {
			m.logger.Error("Error building layer",
				zap.String("layer", name),
				zap.Error(err))
			return err
		}

	}

	err := redisClient.Flush()
	if err != nil {
		return errors.Wrap(err, "error flushing commands")
	}

	// parse all responses
	// tODO here we do not need to lock cause there's only one thread
	for _, name := range m.layerNames {
		layer, err := m.buildLayer(redisClient, name)
		if err != nil {
			return errors.Wrapf(err, "error loading layer %s", name)
		}
		m.StateDict[name].mu.Lock()
		m.StateDict[name].layer = layer
		m.StateDict[name].mu.Unlock()
	}

	return nil
}

// Clear wipes the statedict of the model
func (m *Model) Clear() {
	m.StateDict = make(map[string]*Entry)

	// initialize with nil parameters
	for _, name := range m.layerNames {
		m.StateDict[name] = &Entry{mu: sync.Mutex{}}
	}
	m.logger.Debug("Wiped model state")
}

// Summary runs through the layers of a model and prints its info
func (m *Model) Summary() {
	for name, entry := range m.StateDict {
		m.logger.Info("Layer",
			zap.String("name", name),
			zap.Any("shape", entry.layer.Weights.Shape()),
		)
	}

}

// Save saves the new updated weights and bias in the database so it can be retrieved
// by the following functions
func (m *Model) Save() error {
	m.logger.Info("Publishing model on the database")

	// get the client
	redisClient := util.GetRedisAIClient(m.redisPool, true)
	defer redisClient.Close()

	// start the transaction in the redis client
	redisClient.DoOrSend("MULTI", nil, nil)
	for name, entry := range m.StateDict {
		m.logger.Debug("Setting layer", zap.String("name", name))
		err := m.setLayer(redisClient, name, entry.layer)
		if err != nil {
			return err
		}
	}

	// execute all commands as a batch and empty response buffer
	_, err := redisClient.ActiveConn.Do("EXEC")
	if err != nil {
		return errors.Wrap(err, "could not save tensors")
	}

	m.logger.Info("Model published in the DB")
	return nil

}

// SetLayer saves a layer's weights and bias if available in the storage
func (m *Model) setLayer(redisClient *redisai.Client, name string, layer *Layer) error {

	err := m.setWeights(redisClient, name, layer)
	if err != nil {
		return err
	}

	return nil
}

func (m *Model) setWeights(redisClient *redisai.Client, name string, layer *Layer) error {
	args, _ := makeArgs(m.jobId, name, layer.Weights.Shape(), layer.Dtype, layer.Weights.Data())
	_, err := redisClient.DoOrSend("AI.TENSORSET", *args, nil)
	if err != nil {
		return errors.Wrapf(err, "could not set weights of layer %v", name)
	}
	return nil
}

// fetchLayer calls the tensor get function in the pipelined client
// the results are pipelined and are thus gathered later on in the buildLayer
// function
func (m *Model) fetchLayer(redisClient *redisai.Client, name string, funcId int) error {

	// call get blob but ignore the results cause those are pipelined
	tensorName := getWeightKeys(name, m.jobId, funcId)
	_, _, _, err := redisClient.TensorGetBlob(tensorName)
	if err != nil {
		return err
	}

	return nil

}

// buildLayer reads the pipelined response, parses it and returns the Layer
func (m *Model) buildLayer(redisClient *redisai.Client, name string) (*Layer, error) {

	// get the next response in the pipelined client
	resp, err := redisClient.Receive()
	err, dtype, shapeInt64, blob := redisai.ProcessTensorGetReply(resp, err)
	if err != nil {
		return nil, err
	}

	switch dtype {
	case redisai.TypeFloat32:
		values, err := blobToFloatArray(blob.([]byte), shapeInt64)
		if err != nil {
			return nil, err
		}
		shapeInt := shapeToIntArray(shapeInt64...)

		t := tensor.New(tensor.WithShape(shapeInt...), tensor.WithBacking(values))

		return &Layer{
			Name:    name,
			Dtype:   dtype,
			Weights: t,
		}, nil

	case redisai.TypeInt64:
		values, err := blobtoIntArray(blob.([]byte), shapeInt64)
		if err != nil {
			return nil, err
		}
		shapeInt := shapeToIntArray(shapeInt64...)

		t := tensor.New(tensor.WithShape(shapeInt...), tensor.WithBacking(values))

		return &Layer{
			Name:    name,
			Dtype:   dtype,
			Weights: t,
		}, nil

	default:
		m.logger.Error("Unknown datatype for tensor",
			zap.String("dtype", dtype))
		return nil, errors.New("Unkown datatype for tensor")
	}

}

// shuffleLayers randomly shuffles the layer names so they can be loaded in a different
// order thus increasing concurrency
func shuffleLayers(layers []string, seed int) []string {

	// set random seed so they're different
	rand.Seed(int64(seed))

	// copy the original to avoid race
	names := make([]string, len(layers))
	copy(names, layers)

	// shuffle the names
	rand.Shuffle(len(names), func(i, j int) {
		names[i], names[j] = names[j], names[i]
	})

	return names
}

// Update fetches the layers saved by a function and adds them to the statedict
func (m *Model) Update(funcId int) {

	m.logger.Debug("Updating model layers",
		zap.Int("funcId", funcId))

	redisClient := util.GetRedisAIClient(m.redisPool, true)
	defer redisClient.Close()

	// if we load the layers in the same order in all workers,
	// we would end up with the same performance of having to wait for each other
	// instead randomly iterate the map. Use the function id as the random seed
	layers := shuffleLayers(m.layerNames, funcId)

	// fetch layer by name in non-deterministic order
	// by iterating the shuffled layers
	for _, layer := range layers {
		err := m.fetchLayer(redisClient, layer, funcId)
		if err != nil {
			m.logger.Error("could not fetch layer",
				zap.Error(err),
				zap.String("name", layer),
				zap.Int("funcId", funcId))
			return
		}
	}

	redisClient.Flush()

	for _, layerName := range layers {
		layer, err := m.buildLayer(redisClient, layerName)
		if err != nil {
			m.logger.Error("Could not build layer from database",
				zap.Error(err),
				zap.String("name", layerName),
				zap.Int("funcId", funcId))
			return
		}

		// lock the entry
		m.StateDict[layerName].mu.Lock()
		if m.StateDict[layerName].layer == nil {
			m.StateDict[layerName].layer = layer
		} else {
			total := m.StateDict[layerName].layer
			total.Weights, err = total.Weights.Add(layer.Weights)
			if err != nil {
				m.logger.Error("Error adding weights",
					zap.Error(err))
			}
		}
		m.StateDict[layerName].mu.Unlock()
	}

	m.logger.Debug("Model updated",
		zap.Int("funcId", funcId))

}
