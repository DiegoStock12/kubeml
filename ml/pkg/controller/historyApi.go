package controller

import (
	"context"
	"encoding/json"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"github.com/gorilla/mux"
	"go.mongodb.org/mongo-driver/bson"
	"go.uber.org/zap"
	"net/http"
)

// listHistories returns a list of the histories in the database
func (c *Controller) listHistories(w http.ResponseWriter, r *http.Request) {

	c.logger.Debug("Listing histories")

	var histories []api.History
	collection := c.mongoClient.Database("kubeml").Collection("history")
	//opts := options.Find().SetProjection(bson.M{"_id":1, "task":1})
	cursor, err := collection.Find(context.TODO(), bson.M{})
	if err != nil {
		c.logger.Error("Could not get document lists", zap.Error(err))
		http.Error(w, "Could not get document lists", http.StatusNotFound)
		return
	}

	err = cursor.All(context.TODO(), &histories)
	if err != nil {
		c.logger.Error("could not extract histories from cursor", zap.Error(err))
		http.Error(w, "error processing request", http.StatusInternalServerError)
		return
	}

	resp, err := json.Marshal(histories)
	if err != nil {
		c.logger.Error("Could not parse json histories", zap.Error(err))
		http.Error(w, "error processing request", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(resp)

}

// getHistory gets a history from mongoDB
func (c *Controller) getHistory(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	taskId := vars["taskId"]

	c.logger.Debug("Getting history", zap.String("taskId", taskId))

	// Use the mongo client to get the history
	var history api.History
	collection := c.mongoClient.Database("kubeml").Collection("history")
	err := collection.FindOne(context.TODO(), bson.M{"_id": taskId}).Decode(&history)
	if err != nil {
		c.logger.Error("Could not find history",
			zap.Error(err))
		http.Error(w, "Could not find history for request", http.StatusNotFound)
		return
	}

	resp, err := json.MarshalIndent(history, "", "  ")
	if err != nil {
		c.logger.Error("Could not marshal history",
			zap.Error(err))
		http.Error(w, "Error marshaling request", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(resp)
}

// deleteHistory deletes a training history from the database given its ID
func (c *Controller) deleteHistory(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	taskId := vars["taskId"]

	c.logger.Debug("Deleting history", zap.String("taskId", taskId))

	collection := c.mongoClient.Database("kubeml").Collection("history")
	_, err := collection.DeleteOne(context.TODO(), bson.M{"_id": taskId}, nil)
	if err != nil {
		c.logger.Error("Could not find history", zap.Error(err))
		http.Error(w, "Could not find history to delete", http.StatusNotFound)
		return
	}

	w.WriteHeader(http.StatusOK)
}

// pruneHistories deletes all the histories in the database
func (c *Controller) pruneHistories(w http.ResponseWriter, r *http.Request) {

	c.logger.Debug("Deleting all histories")

	collection := c.mongoClient.Database("kubeml").Collection("history")
	err := collection.Drop(context.Background())
	if err != nil {
		c.logger.Error("Could not delete histories", zap.Error(err))
		http.Error(w, "Could not delete histories", http.StatusNotFound)
		return
	}

	w.WriteHeader(http.StatusOK)
}
