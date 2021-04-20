package train

import (
	"github.com/diegostock12/kubeml/ml/pkg/api"
	kerror "github.com/diegostock12/kubeml/ml/pkg/error"
	"github.com/diegostock12/kubeml/ml/pkg/util"
	"github.com/pkg/errors"
	"go.uber.org/zap"
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

	// read the layer name array from the response
	layers, err := parseLayerNames(resp)
	if err != nil {
		return nil, errors.Wrap(err, "could not read layer names")
	}

	return layers, nil

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
		go job.launchFunction(i, funcUrl, Train, wg, respChan, errChan)
	}
	wg.Wait()

	// check that at least some functions returned without errors
	if err := job.checkFunctionErrors(respChan, errChan); err != nil {
		return 0, nil, err
	}

	// get the average loss
	loss, funcs := getAverageLoss(respChan)

	return loss, funcs, nil
}

// invokeValFunctions After getting all the gradients and publishing the new model invoke
// the validations functions to get the performance of the system, these are returned as a dict
// containing the accuracy, loss and number of datapoints processed by each of the functions.
//
// Returns the accuracy and loss of the functions
func (job *TrainJob) invokeValFunctions() (float64, float64, error) {

	wg := &sync.WaitGroup{}
	respChan := make(chan *FunctionResults, job.parallelism)
	errChan := make(chan error, job.parallelism)

	for i := 0; i < job.parallelism; i++ {
		wg.Add(1)
		job.logger.Debug("Invoking validation function", zap.Int("id", i))
		args := FunctionArgs{Id: i, Num: job.parallelism}
		funcUrl := job.buildFunctionURL(args, Validation)
		go job.launchFunction(i, funcUrl, Validation, wg, respChan, errChan)
	}
	wg.Wait()

	// check that at least some functions returned without errors
	if err := job.checkFunctionErrors(respChan, errChan); err != nil {
		return 0, 0, err
	}

	accuracy, loss, total := getValidationMetrics(respChan)

	// Update the history with the new results
	job.logger.Debug("Got validation results",
		zap.Float64("accuracy", accuracy),
		zap.Float64("loss", loss),
		zap.Float64("total points", total))

	return accuracy, loss, nil

}

// launchFunction launches a training function and sends the results to the
// invokeTrainFunctions function. Which averages the results and adds them to the history
func (job *TrainJob) launchFunction(
	funcId int,
	funcUrl string,
	task FunctionTask,
	wg *sync.WaitGroup,
	respChan chan *FunctionResults,
	errChan chan error) {

	// If the functions are Training, we need to perform
	// extra actions for the k-avg algorithm to know when to sync,
	// if we are validating we skip this
	if task == Train {
		defer func() {
			// Send the finish notification and update the model
			job.finishCh <- &finishNotification{funcId: funcId}
			job.model.Update(funcId)

			job.logger.Debug("adding 1 to the finished functions")
			atomic.AddInt64(&job.finishedFuncs, 1)
			job.wgIteration.Done()
		}()
	}

	defer wg.Done()

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

	respChan <- &FunctionResults{
		funcId:  funcId,
		results: res,
	}

}
