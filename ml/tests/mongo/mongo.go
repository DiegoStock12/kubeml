package main

import (
	"context"
	"fmt"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// Mongo address in minikube
const (
	MONGO_IP = "192.168.99.101"
	MONGO_PORT = 30933
	DB_NAME = "test"
)

type Person struct {
	Name string
	Age int
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

	collection := client.Database("test").Collection("example")
	_, err = collection.DeleteMany(context.TODO(), bson.M{})
	fmt.Println(collection.EstimatedDocumentCount(context.TODO(), nil))

	d := Person{
		Name: "Diego",
		Age:  22,
	}
	//t := Person{
	//	Name: "Tomasz",
	//	Age:  25,
	//}
	//a := Person{
	//	name: "Alvaro",
	//	age:  22,
	//}

	_, err= collection.InsertOne(context.TODO(), d)
	panicIf(err)


	var list []Person
	cursor, err := collection.Find(context.TODO(), bson.M{})
	err = cursor.All(context.TODO(), &list)
	panicIf(err)
	fmt.Println(list)




}