package main

import (
	"context"
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
	DB_NAME    = "kubeml"
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
	defer client.Disconnect(context.TODO())

	err = client.Connect(context.TODO())
	panicIf(err)

	err = client.Ping(context.TODO(), nil)
	panicIf(err)

	opts := options.Find().SetProjection(bson.M{"_id":1, "task":1})
	collection := client.Database("kubeml").Collection("history")
	cursor, err := collection.Find(context.TODO(), bson.M{}, opts)
	panicIf(err)

	var ids []api.History
	err = cursor.All(context.TODO(), &ids)
	panicIf(err)


	fmt.Println(ids)


}
