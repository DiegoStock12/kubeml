package scheduler

import (
	"encoding/json"
	"fmt"
	"github.com/gorilla/mux"
	"go.uber.org/zap"
	"io/ioutil"
	"net/http"
)



// Receives the layerNames as a JSON array
func (ps *ParameterServer) handleLayerNames(w http.ResponseWriter, r *http.Request)  {

	// Gets the layerNames in the request body instead of in the database
	var names []string

	data, err := ioutil.ReadAll(r.Body)
	if err != nil {
		ps.logger.Error("Error parsing the body", zap.Error(err))
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	ps.logger.Debug("Received the following JSON", zap.String("json", string(data)))

	if err = json.Unmarshal(data, &names); err != nil {
		ps.logger.Error("Could not parse the JSON data", zap.Error(err), zap.String("data", string(data)))
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	ps.logger.Debug("Parsed the layer names", zap.Any("names", names))
	w.WriteHeader(http.StatusOK)

	// TODO maybe this is not needed and we can get the result already from the http request
	ps.layerChan <- names
}

// Invoked by the serverless functions when they finish an epoch, should update the model
func (ps *ParameterServer) handleFinish(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	funcId := vars["funcId"]

	ps.logger.Info("Received finish signal from function, updating model",
		zap.String("funcId", funcId))

	w.WriteHeader(http.StatusOK)

	// Update the model with the new gradients
	err := ps.model.Update(funcId)
	if err != nil {
		ps.logger.Error("Error while updating model",
			zap.Error(err))
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	ps.logger.Info("Updated model weights")

	// Change atomically the number of tasks to finish
	// If there are no other tasks tell to the PS that it should
	// Start the next epoch
	ps.numLock.Lock()
	defer ps.numLock.Unlock()
	ps.toFinish -=1
	if ps.toFinish == 0{
		ps.epochChan <- struct{}{}
	}

}

// Returns the handler for calls from the functions
func (ps *ParameterServer) GetHandler() http.Handler {
	r := mux.NewRouter()
	r.HandleFunc("/finish/{funcId}", ps.handleFinish).Methods("POST")
	r.HandleFunc("/layers",ps.handleLayerNames).Methods("POST")

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
