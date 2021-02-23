package main

import (
	"context"
	"fmt"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"log"
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



	res, err := client.ListDatabases(context.TODO(), bson.M{}, &options.ListDatabasesOptions{})
	panicIf(err)
	for _, d := range res.Databases {
		fmt.Println(d.Name, d.SizeOnDisk)
	}

	cols, err := client.Database("kubeml").ListCollections(context.TODO(), bson.M{}, &options.ListCollectionsOptions{})
	panicIf(err)

	for cols.Next(context.TODO()) {
		var col bson.M
		if err := cols.Decode(&col); err != nil {
			log.Fatal(err)
		}

		fmt.Println(col["name"])
		fmt.Println(col["info"])
		for k := range col {
			fmt.Println(k)
		}
	}

	collection := client.Database("kubeml").Collection("history")
	count, err := collection.EstimatedDocumentCount(context.TODO(), nil)
	panicIf(err)
	fmt.Println("count is", count)
	cursor, err := collection.Find(context.TODO(), bson.M{}, nil)
	panicIf(err)

	var ids []api.History
	err = cursor.All(context.TODO(), &ids)
	panicIf(err)
	//
	//
	fmt.Println(ids)

}
