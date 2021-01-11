package main

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// Mongo address in minikube
const (
	MONGO_IP   = "192.168.99.101"
	MONGO_PORT = 30933
	DB_NAME    = "test"
)

type Person struct {
	Name string
	Age  int
}

func panicIf(err error) {
	if err != nil {
		panic(err)
	}
}

func createMongoURI() string {
	return fmt.Sprintf("mongodb://%s:%d", MONGO_IP, MONGO_PORT)
}

func main() {

	uri := createMongoURI()
	fmt.Println(uri)
	client, err := mongo.NewClient(options.Client().ApplyURI(uri))
	panicIf(err)

	err = client.Connect(context.TODO())
	panicIf(err)

	err = client.Ping(context.TODO(), nil)
	panicIf(err)

	// try to create a history with data and so on
	history := map[string][]interface{}{
		"parallelism": {1, 2, 3},
		"loss":        {1.0, 0.999, 7.89},
		"accuracy":    {0.99, 0.76, 0.87},
	}

	fmt.Println(history)

	history["loss"] = append(history["loss"], 1.87, 6.87)

	fmt.Println(history)

	collection := client.Database("kubeml").Collection("history")
	//_, err= collection.InsertOne(context.TODO(), history)
	//panicIf(err)
	//
	var h api.History
	err = collection.FindOne(context.TODO(), bson.M{"_id": "d5f366cb"}).Decode(&h)
	panicIf(err)

	pretty, _ := json.MarshalIndent(h, "", " ")
	fmt.Println(string(pretty))

}
