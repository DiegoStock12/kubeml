package train

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/diegostock12/kubeml/ml/pkg/util"
	"github.com/gomodule/redigo/redis"
	"github.com/pkg/errors"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"
)

func createMongoURI() string {
	if util.IsDebugEnv() {
		return api.MongoUrlDebug
	} else {
		return fmt.Sprintf("mongodb://%s:%d", api.MongoUrl, api.MongoPort)
	}
}

// getAverageLoss iterates through the function results gotten from several
// training functions and returns the average loss and the ids of the functions that completed
func getAverageLoss(respChan chan *FunctionResults) (float64, []int) {
	var funcs []int
	var loss float64

	// close the channel so it can be iterated over
	close(respChan)
	for response := range respChan {
		loss += response.results["loss"]
		funcs = append(funcs, response.funcId)
	}

	avgLoss := loss / float64(len(funcs))
	return avgLoss, funcs
}

// parseFunctionResults takes care of extracting the results from the response body
func parseFunctionResults(resp *http.Response) (map[string]float64, error) {

	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, "unable to read response body")
	}

	var results map[string]float64
	err = json.Unmarshal(body, &results)
	if err != nil {
		return nil, err
	}

	return results, nil
}

// parseResponseError gets the error resulting from the function calls
// ans extracts it from the response
func parseResponseError(data []byte) (funcError error, err error) {
	var resp map[string]interface{}
	err = json.Unmarshal(data, &resp)
	if err != nil {
		return nil, err
	}

	errMsg, exists := resp["error"]
	if !exists {
		return nil, errors.New("could not find error message in response")
	}

	return errors.New(errMsg.(string)), nil

}

// lastValue simply returns the last value of the array if not empty,
// if empty return 0
func lastValue(arr []float64) float64 {
	if len(arr) == 0 {
		return 0
	}
	return arr[len(arr)-1]
}

// getLatestMetrics gets the last entry of the history and returns a metrics
// object that will be sent to the parameter server api to update the counters
// of the job
func getLatestMetrics(history *api.JobHistory) *api.MetricUpdate {
	return &api.MetricUpdate{
		ValidationLoss: lastValue(history.ValidationLoss),
		Accuracy:       lastValue(history.Accuracy),
		TrainLoss:      lastValue(history.TrainLoss),
		Parallelism:    lastValue(history.Parallelism),
		EpochDuration:  lastValue(history.EpochDuration),
	}
}

// clearTensors simply drops the keys and values used during training by the
// different functions and keeps only the reference model in the database
// to save space
func (job *TrainJob) clearTensors() {

	filterStr := fmt.Sprintf("%s*/*", job.jobId)
	tensorListArgs := redis.Args{filterStr}
	tensorNames, err := redis.Strings(job.redisClient.DoOrSend("KEYS", tensorListArgs, nil))
	if err != nil {
		job.logger.Error("Error accessing tensors to be deleted", zap.Error(err))
		return
	}

	if len(tensorNames) == 0 {
		job.logger.Error("No tensors found in storage")
		return
	}

	// delete the temporary tensors in one call
	deleteArgs := redis.Args{}.AddFlat(tensorNames)
	num, err := redis.Int(job.redisClient.DoOrSend("DEL", deleteArgs, nil))
	if err != nil {
		job.logger.Error("Error deleting database tensors", zap.Error(err))
		return
	}
	if num == 0 {
		job.logger.Warn("No tensors with this name found in the database")
	}
	job.logger.Debug("Delete from the database", zap.Int("num tensors", num))
}

// saveTrainingHistory saves the history in the mongo database
func (job *TrainJob) saveTrainingHistory() {
	// get the mongo connection
	client, err := mongo.NewClient(options.Client().ApplyURI(createMongoURI()))
	if err != nil {
		job.logger.Error("Could not create mongo client", zap.Error(err))
		return
	}

	// Save the history in the kubeml database in the history collections
	err = client.Connect(context.TODO())
	if err != nil {
		job.logger.Error("Could not connect to mongo", zap.Error(err))
		return
	}
	defer client.Disconnect(context.TODO())

	// Create the history and index by id
	collection := client.Database("kubeml").Collection("history")
	h := api.History{
		Id:   job.jobId,
		Task: job.task.Parameters,
		Data: job.history,
	}

	// insert it in the DB
	resp, err := collection.InsertOne(context.TODO(), h)
	if err != nil {
		job.logger.Error("Could not insert the history in the database",
			zap.Error(err))
	}

	job.logger.Info("Inserted history", zap.Any("id", resp.InsertedID))

}
