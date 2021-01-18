package controller

import (
	"fmt"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/diegostock12/thesis/ml/pkg/util"
	"go.uber.org/zap"
	"net/http"
	"net/http/httputil"
	"net/url"
)



// StorageServiceProxy returns the reverse proxy that the controller
// uses to redirect all the storage uploads and deletions to the storage service
func (c *Controller) StorageServiceProxy(w http.ResponseWriter, r *http.Request)  {
	var ssUrl *url.URL
	var err error
	if util.IsDebugEnv() {
		ssUrl, err = url.Parse(api.STORAGE_ADDRESS_DEBUG)
	} else {
		ssUrl, err = url.Parse(api.STORAGE_ADDRESS)
	}
	if err != nil {
		c.logger.Error("Error parsing url",
			zap.Error(err),
			zap.String("url", api.STORAGE_ADDRESS_DEBUG))
		http.Error(w, fmt.Sprintf("Error parsing url %s: %v", api.STORAGE_ADDRESS, err),
			http.StatusInternalServerError)
		return
	}

	// create a director function that performs the necessary changes
	// so the request can be redirected to the appropriate address of the
	// storage service
	director := func(req *http.Request) {
		req.URL.Scheme = ssUrl.Scheme
		req.URL.Host = ssUrl.Host
		req.Host = ssUrl.Host
	}

	proxy := &httputil.ReverseProxy{
		Director: director,
	}

	proxy.ServeHTTP(w, r)


}
