package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/diegostock12/kubeml/ml/pkg/util"
	"github.com/gorilla/mux"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.uber.org/zap"
	"net/http"
	"net/http/httputil"
	"net/url"
)

// defaultBatchSize is the default groups of samples in each document.
// this split is done at upload time
const defaultBatchSize int64 = 64
const CollectionTrain = "train"
const CollectionTest = "test"

// defaultDatabases shows the admin or non-dataset databases that we will
// omit when returning the list of datasets
var defaultDatabases = map[string]struct{}{
	"admin":  {},
	"config": {},
	"kubeml": {},
	"local":  {},
}

// storageServiceProxy returns the reverse proxy that the controller
// uses to redirect all the storage uploads and deletions to the storage service
func (c *Controller) storageServiceProxy(w http.ResponseWriter, r *http.Request) {
	var ssUrl *url.URL
	var err error
	if util.IsDebugEnv() {
		ssUrl, err = url.Parse(api.STORAGE_ADDRESS_DEBUG)
	} else {
		ssUrl, err = url.Parse(api.STORAGE_ADDRESS)
	}
	if err != nil {
		c.logger.Error("Error parsing url",
			zap.Error(err),
			zap.String("url", api.STORAGE_ADDRESS_DEBUG))
		http.Error(w, fmt.Sprintf("Error parsing url %s: %v", api.STORAGE_ADDRESS, err),
			http.StatusInternalServerError)
		return
	}

	// create a director function that performs the necessary changes
	// so the request can be redirected to the appropriate address of the
	// storage service
	director := func(req *http.Request) {
		req.URL.Scheme = ssUrl.Scheme
		req.URL.Host = ssUrl.Host
		req.Host = ssUrl.Host
	}

	proxy := &httputil.ReverseProxy{
		Director: director,
	}

	proxy.ServeHTTP(w, r)

}

// getDataset returns the summary of a dataset
func (c *Controller) getDataset(w http.ResponseWriter, r *http.Request) {

	vars := mux.Vars(r)
	datasetName := vars["name"]

	c.logger.Debug("getting dataset")

	results, err := c.mongoClient.ListDatabases(context.Background(), bson.M{}, &options.ListDatabasesOptions{})
	if err != nil {
		c.logger.Error("error getting list of databases",
			zap.Error(err))
		http.Error(w, "error getting list of databases", http.StatusInternalServerError)
		return
	}

	for _, dataset := range results.Databases {
		if _, isDefaultDatabase := defaultDatabases[dataset.Name]; !isDefaultDatabase && datasetName == dataset.Name {
			summary := api.DatasetSummary{
				Name: dataset.Name,
			}

			// get the train and test collections and their size
			trainCollection := c.mongoClient.Database(dataset.Name).Collection(CollectionTrain)
			count, err := trainCollection.EstimatedDocumentCount(context.Background(), nil)
			if err != nil {
				c.logger.Error("error counting documents of collection",
					zap.String("dataset", dataset.Name),
					zap.String("collection", CollectionTrain))
			} else {
				summary.TrainSetSize = ((count * defaultBatchSize) / 100) * 100
			}

			testCollection := c.mongoClient.Database(dataset.Name).Collection(CollectionTest)
			count, err = testCollection.EstimatedDocumentCount(context.Background(), nil)
			if err != nil {
				c.logger.Error("error counting documents of collection",
					zap.String("dataset", dataset.Name),
					zap.String("collection", CollectionTest))
			} else {
				summary.TestSetSize = ((count * defaultBatchSize) / 100) * 100
			}

			resp, err := json.Marshal(summary)
			if err != nil {
				c.logger.Error("error marshaling dataset data",
					zap.Error(err))
				http.Error(w, "error marshaling response", http.StatusInternalServerError)
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write(resp)
			return
		}
	}

	w.WriteHeader(http.StatusNotFound)
	http.Error(w, "dataset not found", http.StatusNotFound)

}

// listDatasets returns the summaries of all the datasets
func (c *Controller) listDatasets(w http.ResponseWriter, r *http.Request) {

	c.logger.Debug("Listing datasets")

	var datasets []api.DatasetSummary
	results, err := c.mongoClient.ListDatabases(context.Background(), bson.M{}, &options.ListDatabasesOptions{})
	if err != nil {
		c.logger.Error("error getting list of databases",
			zap.Error(err))
		http.Error(w, "error getting list of databases", http.StatusInternalServerError)
		return
	}

	// iterate the databases and create the return object.
	// check if the dataset belongs to the admin datasets and omit it
	// if that's the case
	for _, dataset := range results.Databases {
		if _, isDefaultDatabase := defaultDatabases[dataset.Name]; !isDefaultDatabase {
			summary := api.DatasetSummary{
				Name: dataset.Name,
			}

			// get the train and test collections and their size
			trainCollection := c.mongoClient.Database(dataset.Name).Collection(CollectionTrain)
			count, err := trainCollection.EstimatedDocumentCount(context.Background(), nil)
			if err != nil {
				c.logger.Error("error counting documents of collection",
					zap.String("dataset", dataset.Name),
					zap.String("collection", CollectionTrain))
				continue
			}
			summary.TrainSetSize = ((count * defaultBatchSize) / 100) * 100

			testCollection := c.mongoClient.Database(dataset.Name).Collection(CollectionTest)
			count, err = testCollection.EstimatedDocumentCount(context.Background(), nil)
			if err != nil {
				c.logger.Error("error counting documents of collection",
					zap.String("dataset", dataset.Name),
					zap.String("collection", CollectionTest))
				continue
			}
			summary.TestSetSize = ((count * defaultBatchSize) / 100) * 100

			datasets = append(datasets, summary)
		}
	}

	resp, err := json.Marshal(datasets)
	if err != nil {
		c.logger.Error("error marshaling dataset data",
			zap.Error(err))
		http.Error(w, "error marshaling response", http.StatusInternalServerError)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(resp)

}
