package v1

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/pkg/errors"
	"io"
	"io/ioutil"
	"mime/multipart"
	"net/http"
	"os"
)

var (
	filenames = []string{"x-train", "y-train", "x-test", "y-test"}
)

type (

	// DatasetsGetter returns an object to interact with the
	// kubeml datasets
	DatasetsGetter interface {
		Datasets() DatasetInterface
	}

	// DatasetInterface has methods to work with dataset resources
	DatasetInterface interface {
		Create(name, trainData, trainLabels, testData, testLabels string) error
		Delete(name string) error
		Get(name string) (*api.DatasetSummary, error)
		List() ([]api.DatasetSummary, error)
	}

	// datasets implements DatasetInterface
	datasets struct {
		controllerUrl string
		httpClient    *http.Client
	}
)

func newDatasets(c *V1) DatasetInterface {
	return &datasets{
		controllerUrl: c.controllerUrl,
		httpClient:    c.httpClient,
	}
}

func (d *datasets) Create(name, trainData, trainLabels, testData, testLabels string) error {
	url := d.controllerUrl + "/dataset/" + name

	// Create the files to index the file name
	files := []string{trainData, trainLabels, testData, testLabels}

	body := new(bytes.Buffer)
	writer := multipart.NewWriter(body)

	// For each of the files add a multipart form field
	// with the specific filename and its contents
	for i, name := range files {
		file, err := os.Open(name)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("could not open file %s", name))
		}
		defer file.Close()

		part, err := writer.CreateFormFile(filenames[i], file.Name())
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("could not write part from file %s", name))
		}

		_, err = io.Copy(part, file)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("could not copy part from file %s", name))
		}

		fmt.Println("File", name, "added to the multipart form")

	}

	err := writer.Close()
	if err != nil {
		return errors.Wrap(err, "could not close writer")
	}

	resp, err := d.httpClient.Post(url, writer.FormDataContentType(), body)
	if err != nil {
		return errors.Wrap(err, "could not process creation request")
	}
	defer resp.Body.Close()

	var result map[string]string
	respBody, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	err = json.Unmarshal(respBody, &result)

	if resp.StatusCode != http.StatusOK {
		return errors.New(fmt.Sprintf("Could not complete task: %s", result["error"]))
	}

	fmt.Println(result["result"])
	return nil
}

func (d *datasets) Delete(name string) error {
	url := d.controllerUrl + "/dataset/" + name

	req, err := http.NewRequest(http.MethodDelete, url, nil)
	if err != nil {
		return errors.Wrap(err, "could not create request body")
	}

	resp, err := d.httpClient.Do(req)
	if err != nil {
		return errors.Wrap(err, "could not handle request")
	}
	defer resp.Body.Close()

	var result map[string]string
	respBody, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	err = json.Unmarshal(respBody, &result)

	if resp.StatusCode != http.StatusOK {
		return errors.New(fmt.Sprintf("Status code is not OK: %s", result["error"]))
	}

	fmt.Println(result["result"])
	return nil
}

func (d *datasets) Get(name string) (*api.DatasetSummary, error) {
	url := d.controllerUrl + "/dataset/" + name

	resp, err := d.httpClient.Get(url)
	if err != nil {
		return nil, errors.Wrap(err, "could not get perform http request")
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, "could not read responde body")
	}

	var dataset api.DatasetSummary
	err = json.Unmarshal(body, &dataset)
	if err != nil {
		return nil, errors.Wrap(err, "could not decode body")
	}

	return &dataset, nil

}

func (d *datasets) List() ([]api.DatasetSummary, error) {
	url := d.controllerUrl + "/dataset"

	resp, err := d.httpClient.Get(url)
	if err != nil {
		return nil, errors.Wrap(err, "could not get perform http request")
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, "could not read responde body")
	}

	var result []api.DatasetSummary
	err = json.Unmarshal(body, &result)
	if err != nil {
		return nil, errors.Wrap(err, "could not decode body")
	}

	return result, nil
}
