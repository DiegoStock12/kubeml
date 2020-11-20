package scheduler

import (
	"fmt"
	"github.com/gorilla/mux"
	"go.uber.org/zap"
	"net/http"
)

// Invoked by the serverless functions when they finish an epoch, should update the model
func (ps *ParameterServer) handleFinish(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	funcId := vars["funcId"]

	ps.logger.Info("Received finish signal from function, updating model",
		zap.String("funcId", funcId))

	// Update the model with the new gradients
	err := ps.model.Update(ps.psId, funcId)
	if err != nil {
		ps.logger.Error("Error while updating model",
			zap.Error(err))
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	ps.logger.Info("Updated model weights")
}

// Returns the handler for calls from the functions
func (ps *ParameterServer) GetHandler() http.Handler {
	r := mux.NewRouter()
	r.HandleFunc("/notifyFinish/{funcId}", ps.handleFinish).Methods("POST")

	return r
}

// Start the API at the given port
func (ps *ParameterServer) Serve(port int) {
	ps.logger.Info("Starting Parameter Server api",
		zap.Int("port", port),
		zap.String("psId", ps.psId))

	addr := fmt.Sprintf(":%v", port)

	// Start serving the endpoint
	err := http.ListenAndServe(addr, ps.GetHandler())
	ps.logger.Fatal("Parameter Server API done",
		zap.String("psID", ps.psId),
		zap.Error(err))
}
