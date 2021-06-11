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
	"time"
)

// updateValidationMetrics updates the validation statistics in the PS
func (job *TrainJob) updateValidationMetrics(valLoss, accuracy float64) error {
	job.history.ValidationLoss = append(job.history.ValidationLoss, valLoss)
	job.history.Accuracy = append(job.history.Accuracy, accuracy)

	// send the update to the PS
	err := job.ps.UpdateMetrics(job.jobId, getLatestMetrics(&job.history))
	if err != nil {
		return errors.Wrap(err, "error sending validation update to parameter server")
	}

	return nil

}

// updateTrainMetrics updates the metrics in the job history and sends an update to the
// parameter server to publish the new metrics to prometheus
func (job *TrainJob) updateTrainMetrics(loss float64, elapsed time.Duration) error {

	// add the new metrics to the history
	job.history.Parallelism = append(job.history.Parallelism, float64(job.parallelism))
	job.history.EpochDuration = append(job.history.EpochDuration, elapsed.Seconds())
	job.history.TrainLoss = append(job.history.TrainLoss, loss)

	// send the update to the PS
	err := job.ps.UpdateMetrics(job.jobId, getLatestMetrics(&job.history))
	if err != nil {
		return errors.Wrap(err, "error sending train update to parameter server")
	}

	return nil
}

func createMongoURI() string {
	if util.IsDebugEnv() {
		return api.MongoUrlDebug
	} else {
		return fmt.Sprintf("mongodb://%s:%d", api.MongoUrl, api.MongoPort)
	}
}

//parseLayerNames is used by the init function to parse the array of layer names
// sent by the init function in the severless function. Theses names will allow the job to load the model layers
func parseLayerNames(resp *http.Response) ([]string, error) {
	var names []string

	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, "could not read body")
	}

	err = json.Unmarshal(body, &names)
	if err != nil {
		return nil, errors.Wrap(err, "error unmarshaling json")
	}

	return names, nil

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

// getValidationMetrics analyzes the results of validation functions containing
// the accuracy, the loss and the number of datapoints used in each, and performs
// the weighted averaging of both according to the number of points
func getValidationMetrics(respChan chan *FunctionResults) (float64, float64, float64) {
	var accuracy float64
	var loss float64
	var total float64

	// close the channel
	close(respChan)

	// the json has atributes loss, accuracy and length
	for response := range respChan {
		length := response.results["length"]
		loss += response.results["loss"] * length
		accuracy += response.results["accuracy"] * length
		total += length
	}

	// divide by the total number of points to get the accuracy
	accuracy /= total
	loss /= total

	return accuracy, loss, total

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

// checkFunctionErrors checks that all of the functions or some of them returned without
// errors
func (job *TrainJob) checkFunctionErrors(respChan chan *FunctionResults, errChan chan error) error {

	// based on the number of responses check for the error
	num := len(respChan)
	switch {
	case num == 0:
		select {
		case funcError := <-errChan:
			return errors.Wrap(funcError, "all functions finished with an error")
		default:
			return errors.New("all functions returned an unknown error")
		}

	case num < job.parallelism:
		job.logger.Warn("Some of the functions returned without a result",
			zap.Int("parallelism", job.parallelism),
			zap.Int("responses", num))
		return nil

	}

	return nil
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

	// disable the pipeline in the client
	redisClient := util.GetRedisAIClient(job.redisPool, false)
	defer redisClient.Close()

	// delete all of the tensors for that model in the database
	filterStr := fmt.Sprintf("%s*/*", job.jobId)
	tensorListArgs := redis.Args{filterStr}
	tensorNames, err := redis.Strings(redisClient.DoOrSend("KEYS", tensorListArgs, nil))
	if err != nil {
		job.logger.Error("Error accessing tensors to be deleted", zap.Error(err))
		return
	}

	if len(tensorNames) == 0 {
		job.logger.Error("No tensors found in storage")
		return
	}

	job.logger.Debug("Deleting tensors...", zap.Strings("names", tensorNames))

	// delete the temporary tensors in one call
	deleteArgs := redis.Args{}.AddFlat(tensorNames)
	num, err := redis.Int(redisClient.DoOrSend("DEL", deleteArgs, nil))
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
