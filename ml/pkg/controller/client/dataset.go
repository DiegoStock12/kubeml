package client

import (
	"bytes"
	"fmt"
	"github.com/pkg/errors"
	"io"
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
		defer file.Close()

		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("could not open file %s", name))
		}

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

	resp, err := c.httpClient.Post(url, writer.FormDataContentType(), body)
	if err != nil {
		return errors.Wrap(err, "could not process creation request")
	}

	if resp.StatusCode != http.StatusOK {
		return errors.New(fmt.Sprintf("Could not complete task, result code is %v", resp.StatusCode))
	}

	fmt.Println("Dataset", name, "created succesfully")
	return nil

}

// DeleteDataset deletes a current dataset
func (c *Client) DeleteDataset(name string) error {
	url := c.controllerUrl + "/dataset/" + name

	req, err := http.NewRequest("DELETE", url, nil)
	if err != nil {
		return errors.Wrap(err, "could not create request body")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return errors.Wrap(err, "could not handle request")
	}

	if resp.StatusCode != http.StatusOK {
		return errors.New(fmt.Sprintf("Status code is not OK: %v", resp.StatusCode))
	}

	return nil
}
