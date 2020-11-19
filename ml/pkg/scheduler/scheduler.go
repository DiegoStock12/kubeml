package scheduler

import (
	"github.com/gorilla/mux"
	"net/http"
)

type (
	Scheduler struct {

	}
)

func (s *Scheduler) Start()  {
	r := mux.NewRouter()
	r.HandleFunc("/", )
}

