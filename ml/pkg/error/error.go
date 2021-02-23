package error

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"strings"
)

// Error is the way the API from both the python environment and
// the kubeml components will serialize errors as a JSON response
type Error struct {
	Code    int    `json:"code"`
	Message string `json:"error"`
}

// Error allows the kubeml error to override the default golang error,
// returns the error message
func (e Error) Error() string {
	return e.Message
}

// New creates an error with the http status code passed and
// the error message defined
func New(code int, message string) Error {
	return Error{
		Code:    code,
		Message: message,
	}
}

// CheckFunctionError reads the an object such as a response body
// and returns the error object. If some error happens while reading or
// deserializing it returns said error as the message
func CheckFunctionError(resp *http.Response) error {
	if resp.StatusCode == http.StatusOK {
		return nil
	}

	// if code is not OK, just parse the response body
	defer resp.Body.Close()

	var msg string
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		msg = strings.TrimSpace(string(body))
		return New(resp.StatusCode, msg)
	}

	var funcError Error
	err = json.Unmarshal(body, &funcError)
	if err != nil {
		msg = strings.TrimSpace(string(body))
		return New(resp.StatusCode, msg)
	}

	return funcError
}

// RespondWithError is a convenience function for responding the client with a
// properly formated error
func RespondWithError(w http.ResponseWriter, err Error) {
	body, _ := json.Marshal(err)

	w.WriteHeader(err.Code)
	w.Header().Set("Content-Type", "application/json")
	w.Write(body)
}
