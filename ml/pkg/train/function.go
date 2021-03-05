package train

import (
	"encoding/json"
	"errors"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	kerror "github.com/diegostock12/kubeml/ml/pkg/error"
	"github.com/diegostock12/kubeml/ml/pkg/util"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"sync/atomic"
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
		routerAddr = api.FissionRouterUrlDebug
	} else {
		routerAddr = api.FissionRouterUrl
	}

	values := url.Values{}
	values.Set("task", string(task))
	values.Set("jobId", job.jobId)
	values.Set("N", strconv.Itoa(args.Num))
	values.Set("K", strconv.Itoa(job.K))
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
	funcUrl := job.buildFunctionURL(FunctionArgs{}, Init)
	resp, err := http.Get(funcUrl)
	if err != nil {
		job.logger.Error("Could not call the init function",
			zap.String("funcName", job.task.Parameters.FunctionName),
			zap.Any("request", job.task.Parameters),
			zap.Error(err))

		return nil, err
	}

	// check if an error was returned
	if err = kerror.CheckFunctionError(resp); err != nil {
		return nil, err
	}

	defer resp.Body.Close()
	var names []string
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		job.logger.Fatal("Could not read init function response",
			zap.Error(err))
		return nil, err
	}

	err = json.Unmarshal(body, &names)
	if err != nil {
		job.logger.Error("Could not unmarshall json",
			zap.String("body", string(body)),
			zap.Error(err))
		return names, err
	}

	return names, nil

}

// invokeTrainFunctions Invokes N functions to start the next epoch
// returns the function ids from which it got a response
func (job *TrainJob) invokeTrainFunctions() (float64, []int, error) {

	wg := &sync.WaitGroup{}
	respChan := make(chan *FunctionResults, job.parallelism)
	errChan := make(chan error, job.parallelism)

	for i := 0; i < job.parallelism; i++ {
		wg.Add(1)

		job.logger.Debug("Invoking function", zap.Int("id", i))
		args := FunctionArgs{Id: i, Num: job.parallelism}
		funcUrl := job.buildFunctionURL(args, Train)
		go job.launchFunction(i, funcUrl, wg, respChan, errChan)
	}
	wg.Wait()

	n := len(respChan)
	if n == 0 {
		select {
		case funcError := <-errChan:
			job.logger.Error("All the functions failed with no response",
				zap.Error(funcError))
			return 0, nil, funcError

		default:
			return 0, nil, errors.New("all functions returned an unknown error")

		}

	} else if n != job.parallelism {
		job.logger.Warn("Some of the functions returned without a result",
			zap.Int("parallelism", job.parallelism),
			zap.Int("responses", n))
	}

	// get the average loss
	loss, funcs := getAverageLoss(respChan)

	return loss, funcs, nil
}

// invokeValFunction After getting all the gradients and publishing the new model invoke
// the validation function to get the performance of the system, these are returned as a dict
func (job *TrainJob) invokeValFunction(wg *sync.WaitGroup) {

	defer wg.Done()
	job.logger.Info("Invoking validation function")

	funcUrl := job.buildFunctionURL(FunctionArgs{}, Validation)
	resp, err := http.Get(funcUrl)
	if err != nil {
		job.logger.Error("Could not call the validation function",
			zap.String("funcName", job.task.Parameters.FunctionName),
			zap.Any("request", job.task.Parameters),
			zap.Error(err))
		return
	}

	if err = kerror.CheckFunctionError(resp); err != nil {
		job.logger.Error("validation function returned an error",
			zap.Error(err))
		return
	}

	res, err := parseFunctionResults(resp)
	if err != nil {
		job.logger.Error("could not parse validation results",
			zap.Error(err))
		return
	}

	// Update the history with the new results
	job.logger.Debug("Got validation results", zap.Any("results", res))
	err = job.updateValidationMetrics(res["loss"], res["accuracy"])
	if err != nil {
		job.logger.Error("error sending val results", zap.Error(err))
		return
	}

	job.logger.Debug("History updated", zap.Any("history", job.history))

	// if the accuracy reached the goal, send the notification
	if res["accuracy"] >= job.goalAccuracy {
		job.logger.Debug("goal accuracy reached, sending message",
			zap.Float64("goal", job.goalAccuracy),
			zap.Float64("acc", res["accuracy"]))
		job.accuracyCh <- struct{}{}
	}
}

// launchFunction launches a training function and sends the results to the
// invokeTrainFunctions function. Which averages the results and adds them to the history
func (job *TrainJob) launchFunction(
	funcId int,
	funcUrl string,
	wg *sync.WaitGroup,
	respChan chan *FunctionResults,
	errChan chan error) {

	// after exiting clean the stuff
	defer func() {
		wg.Done()
		job.logger.Debug("adding 1 to the finished functions")
		atomic.AddInt64(&job.finishedFuncs, 1)
		job.wgIteration.Done()
	}()

	resp, err := http.Get(funcUrl)
	if err != nil {
		job.logger.Error("Error when performing request",
			zap.Int("funcId", funcId),
			zap.Error(err))
		errChan <- err
		return
	}

	job.logger.Debug("function finished, checking errors", zap.Int("id", funcId))

	// Check if we got a KubeML error in the response, if so return it in the error chan
	if err = kerror.CheckFunctionError(resp); err != nil {
		job.logger.Debug("returning error...", zap.Error(err))
		errChan <- err
		return
	}

	res, err := parseFunctionResults(resp)
	if err != nil {
		errChan <- err
		return
	}

	job.logger.Info("Sending result to channel and exiting",
		zap.Int("funcId", funcId),
		zap.Any("results", res))

	job.finishCh <- &finishNotification{funcId: funcId}
	respChan <- &FunctionResults{
		funcId:  funcId,
		results: res,
	}

}
