package old

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

// CreateDataset uploads the files to the storage service and
// creates a dataset to be used in future training jobs
func (c *Client) CreateDataset(name, trainData, trainLabels, testData, testLabels string) error {
	url := c.controllerUrl + "/dataset/" + name

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

	resp, err := c.httpClient.Post(url, writer.FormDataContentType(), body)
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

// DeleteDataset deletes a current dataset
func (c *Client) DeleteDataset(name string) error {
	url := c.controllerUrl + "/dataset/" + name

	req, err := http.NewRequest(http.MethodDelete, url, nil)
	if err != nil {
		return errors.Wrap(err, "could not create request body")
	}

	resp, err := c.httpClient.Do(req)
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

// ListDatasets returns a list of the datasets uploaded to kubeml
func (c *Client) ListDatasets() ([]api.DatasetSummary, error) {
	url := c.controllerUrl + "/dataset"

	resp, err := c.httpClient.Get(url)
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
