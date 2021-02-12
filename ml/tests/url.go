package main

import (
	"fmt"
	"github.com/diegostock12/kubeml/ml/pkg/api"
	"net/url"
)

func main(){
	u, _ := url.Parse(api.STORAGE_ADDRESS+"/dataset/test")
	fmt.Println(u.Host, u.Scheme, u.User, u.Path)

}
