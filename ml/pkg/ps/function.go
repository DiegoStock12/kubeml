package ps

import (
	"encoding/json"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/diegostock12/thesis/ml/pkg/util"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"sync"
)

// TODO this should take something to determine the batch of the data that should be used
// buildFunctionURL returns the url that the PS will invoke to execute the function
func (job *TrainJob) buildFunctionURL(funcId, numFunc int, task, funcName string) string {

	var routerAddr string
	if util.IsDebugEnv() {
		routerAddr = api.ROUTER_ADDRESS_DEBUG
	} else {
		routerAddr = api.ROUTER_ADDRESS
	}

	values := url.Values{}
	values.Set("task", task)
	values.Set("psId", job.jobId)
	values.Set("N", strconv.Itoa(numFunc))
	values.Set("funcId", strconv.Itoa(funcId))
	values.Set("batchSize", strconv.Itoa(job.task.Parameters.BatchSize))
	values.Set("lr", strconv.FormatFloat(float64(job.task.Parameters.LearningRate), 'f', -1, 32))

	dest := routerAddr + "/" + funcName + "?" + values.Encode()

	job.logger.Debug("Built url", zap.String("url", dest))

	return dest
}

// invokeInitFunction calls a single function which initializes the
// model, saves it to the database and returns the layer names that the job will save
func (job *TrainJob) invokeInitFunction() ([]string, error) {

	job.logger.Info("Invoking init function")
	query := job.buildFunctionURL(0, 1, "init", job.task.Parameters.FunctionName)
	resp, err := http.Get(query)
	defer resp.Body.Close()
	if err != nil {
		// TODO here we should implement retries like in the fetcher specialize in fission
		// TODO maybe create a special function called execute with retries
		job.logger.Error("Could not call the init function",
			zap.String("funcName", job.task.Parameters.FunctionName),
			zap.Any("request", job.task.Parameters),
			zap.Error(err))

		return nil, err
	}

	var names []string
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		job.logger.Fatal("Could not read layer names",
			zap.Error(err))

		return nil, err
	}

	err = json.Unmarshal(data, &names)
	if err != nil {
		job.logger.Error("Could not unmarshall json",
			zap.String("body", string(data)),
			zap.Error(err))
		return names, err
	}

	// Set the layer names
	return names, nil

}

// invokeTrainFunctions Invokes N functions to start the next epoch
// returns the function ids from which it got a response
func (job *TrainJob) invokeTrainFunctions() []int {

	job.logger.Debug("Invoking functions", zap.Int("N", job.parallelism))

	var funcs []int

	// Create the wait group and the channel
	wg := &sync.WaitGroup{}
	respChan := make(chan *funcResults, job.parallelism)

	for i := 0; i < job.parallelism; i++ {
		job.logger.Debug("Invoking function", zap.Int("id", i))
		query := job.buildFunctionURL(i, job.parallelism, "train", job.task.Parameters.FunctionName)

		wg.Add(1)
		// TODO this should return the train accuracy and loss for all of them
		go job.launchFunction(i, query, wg, respChan)
	}

	wg.Wait()

	job.logger.Info("Got all the responses, iterating")
	close(respChan)

	// Calculate the mean and save in the history
	var loss float64
	n := len(respChan)
	if n == 0 {
		job.logger.Fatal("All the functions failed with no response")
	}
	if n != job.parallelism {
		job.logger.Warn("Some of the functions returned without a result",
			zap.Int("parallelism", job.parallelism),
			zap.Int("responses", n))
	}

	// Compute the average loss reported by the functions
	for response := range respChan {
		job.logger.Debug("Got result...", zap.Any("Result", response.results))
		loss += response.results["loss"]
		funcs = append(funcs, response.funcId)
	}
	// After all divide by the number of elements and add to the history
	avgLoss := loss / float64(n)
	job.logger.Info("Epoch had average loss", zap.Float64("loss", avgLoss))
	job.history.TrainLoss = append(job.history.TrainLoss, avgLoss)


	job.logger.Debug("History updated", zap.Any("history", job.history))

	return funcs
}

// invokeValFunction After getting all the gradients and publishing the new model invoke
// the validation function to get the performance of the system, these are returned as a dict
func (job *TrainJob) invokeValFunction(wg *sync.WaitGroup) {

	defer wg.Done()
	job.logger.Info("Invoking validation function")

	// TODO instead of returning the map we could add it to a job level map that tracks the progress
	var results map[string]float64
	query := job.buildFunctionURL(0, 1, "val", job.task.Parameters.FunctionName)
	resp, err := http.Get(query)
	if err != nil {
		// TODO here we should implement retries like in the fetcher specialize in fission
		// TODO maybe create a special function called execute with retries
		job.logger.Error("Could not call the validation function",
			zap.String("funcName", job.task.Parameters.FunctionName),
			zap.Any("request", job.task.Parameters),
			zap.Error(err))
	}
	defer resp.Body.Close()

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		job.logger.Error("Could not read layer names",
			zap.Error(err))

	}

	// Unmarshall the JSON to a dict
	// This JSON should give accuracy, precision, recall...
	err = json.Unmarshal(data, &results)
	if err != nil {
		job.logger.Error("Could not parse JSON",
			zap.String("body", string(data)),
			zap.Error(err))
		return
	}

	job.logger.Debug("Got validation results", zap.Any("results", results))

	// Update the history with the new results
	job.history.ValidationLoss = append(job.history.ValidationLoss, results["loss"])
	job.history.Accuracy = append(job.history.Accuracy, results["accuracy"])
	job.logger.Debug("History updated", zap.Any("history", job.history))

}

// launchFunction launches a training function and sends the results to the
// invokeTrainFunctions function. Which averages the results and adds them to the history
func (job *TrainJob) launchFunction(
	funcId int,
	query string,
	wg *sync.WaitGroup,
	respChan chan *funcResults) {

	job.logger.Info("Starting request for function number", zap.Int("func_id", funcId))
	defer wg.Done()

	// do the request
	resp, err := http.Get(query)
	if err != nil {
		job.logger.Error("Error when performing request",
			zap.Int("funcId", funcId),
			zap.Error(err))
		return
	}
	defer resp.Body.Close()

	var res map[string]float64
	// We get a json with {loss: float64} so parse the json and so on
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		job.logger.Error("Could not read response body", zap.Error(err))
		return
	}

	job.logger.Debug(fmt.Sprintf("Received body, %s", string(body)), zap.Int("funcId", funcId))
	if err = json.Unmarshal(body, &res); err != nil {
		job.logger.Error("Could not parse the JSON data", zap.Error(err),
			zap.String("data", string(body)),
			zap.String("status", resp.Status))
		return
	}

	// send the result to the channel and confirm exit
	job.logger.Info("Sending result to channel and exiting",
		zap.Int("funcId", funcId),
		zap.Any("results", res))
	respChan <- &funcResults{
		funcId:  funcId,
		results: res,
	}

}
