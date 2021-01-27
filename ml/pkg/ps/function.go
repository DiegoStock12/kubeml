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

type (

	// FunctionArgs holds the arguments needed to build
	// the url of a function, such as the function id and
	// parallelism level
	FunctionArgs struct {
		Id  int
		Num int
	}

	// FunctionResults holds the function id and the execution
	// results of a function, be it a training or validation function
	FunctionResults struct {
		funcId  int
		results map[string]float64
	}

	FunctionTask string
)

const (
	Train      FunctionTask = "train"
	Validation FunctionTask = "val"
	Init       FunctionTask = "init"
	Inference  FunctionTask = "infer"
)


// buildFunctionURL returns the url that the PS will invoke to execute the function
func (job *TrainJob) buildFunctionURL(args FunctionArgs, task FunctionTask) string {

	var routerAddr string
	if util.IsDebugEnv() {
		routerAddr = api.ROUTER_ADDRESS_DEBUG
	} else {
		routerAddr = api.ROUTER_ADDRESS
	}

	values := url.Values{}
	values.Set("task", string(task))
	values.Set("psId", job.jobId)
	values.Set("N", strconv.Itoa(args.Num))
	values.Set("funcId", strconv.Itoa(args.Id))
	values.Set("batchSize", strconv.Itoa(job.task.Parameters.BatchSize))
	values.Set("lr", strconv.FormatFloat(float64(job.task.Parameters.LearningRate), 'f', -1, 32))

	dest := routerAddr + "/" + job.task.Parameters.FunctionName + "?" + values.Encode()

	job.logger.Debug("Built url", zap.String("url", dest))

	return dest
}

// invokeInitFunction calls a single function which initializes the
// model, saves it to the database and returns the layer names that the job will save
func (job *TrainJob) invokeInitFunction() ([]string, error) {

	job.logger.Info("Invoking init function")
	query := job.buildFunctionURL(FunctionArgs{}, Init)
	resp, err := http.Get(query)
	if err != nil {
		job.logger.Error("Could not call the init function",
			zap.String("funcName", job.task.Parameters.FunctionName),
			zap.Any("request", job.task.Parameters),
			zap.Error(err))

		return nil, err
	}
	defer resp.Body.Close()

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

	wg := &sync.WaitGroup{}
	respChan := make(chan *FunctionResults, job.parallelism)

	for i := 0; i < job.parallelism; i++ {
		job.logger.Debug("Invoking function", zap.Int("id", i))

		args := FunctionArgs{Id: i, Num: job.parallelism}
		query := job.buildFunctionURL(args, Train)

		wg.Add(1)
		go job.launchFunction(i, query, wg, respChan)
	}

	wg.Wait()
	job.logger.Info("Got all the responses, iterating")
	close(respChan)

	n := len(respChan)
	if n == 0 {
		job.logger.Fatal("All the functions failed with no response")
	}
	if n != job.parallelism {
		job.logger.Warn("Some of the functions returned without a result",
			zap.Int("parallelism", job.parallelism),
			zap.Int("responses", n))
	}

	// Iterate through all the responses from the training functions that are in
	// the channel. Average them onto a single metric which we will publish in the
	// history and also expose to Prometheus
	//
	// It might happen that some of the functions fail. In that case, we count
	// the number of responses and use that number when averaging the layers
	var loss float64
	var funcs []int
	for response := range respChan {
		job.logger.Debug("Got result...", zap.Any("Result", response.results))
		loss += response.results["loss"]
		funcs = append(funcs, response.funcId)
	}

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


	query := job.buildFunctionURL(FunctionArgs{}, Validation)
	resp, err := http.Get(query)
	if err != nil {
		job.logger.Error("Could not call the validation function",
			zap.String("funcName", job.task.Parameters.FunctionName),
			zap.Any("request", job.task.Parameters),
			zap.Error(err))
		return
	}
	defer resp.Body.Close()

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		job.logger.Error("Could not read layer names",
			zap.Error(err))
		return

	}

	var results map[string]float64
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
	respChan chan *FunctionResults) {

	job.logger.Info("Starting request for function number", zap.Int("func_id", funcId))
	defer wg.Done()

	resp, err := http.Get(query)
	if err != nil {
		job.logger.Error("Error when performing request",
			zap.Int("funcId", funcId),
			zap.Error(err))
		return
	}
	defer resp.Body.Close()


	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		job.logger.Error("Could not read response body", zap.Error(err))
		return
	}

	var res map[string]float64
	job.logger.Debug(fmt.Sprintf("Received body, %s", string(body)), zap.Int("funcId", funcId))
	if err = json.Unmarshal(body, &res); err != nil {
		job.logger.Error("Could not parse the JSON data", zap.Error(err),
			zap.String("data", string(body)),
			zap.String("status", resp.Status))
		return
	}

	job.logger.Info("Sending result to channel and exiting",
		zap.Int("funcId", funcId),
		zap.Any("results", res))

	respChan <- &FunctionResults{
		funcId:  funcId,
		results: res,
	}

}
