package cmd

import (
	"fmt"
	fv1 "github.com/fission/fission/pkg/apis/core/v1"
	"github.com/fission/fission/pkg/crd"
	"github.com/hashicorp/go-multierror"
	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"io/ioutil"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"
)

const (
	DefaultNamespace       = metav1.NamespaceDefault
	DefaultEnvironment     = "torch"
	DefaultConcurrency     = 10
	DefaultTimeout         = 500
	DefaultIdleTimeout int = 20
)

var (

	// variables for the create command and delete command
	fnName     string
	fnCodePath string

	functionCmd = &cobra.Command{
		Use:     "function",
		Aliases: []string{"fn"},
		Short:   "Manage Deep Learning deployed functions",
	}

	functionCreateCmd = &cobra.Command{
		Use:   "create",
		Short: "Create a new Deep Learning function",
		RunE:  createFunction,
	}

	functionDeleteCmd = &cobra.Command{
		Use:   "delete",
		Short: "Delete a Deep Learning function",
		RunE:  deleteFunction,
	}

	functionListCmd = &cobra.Command{
		Use:   "list",
		Short: "List deployed Deep Learning functions",
		RunE:  listFunctions,
	}
)

// createFunction creates a new function
// this process entails namely 3 steps
//
// 1. Read the source code file and create a fission package with it,
// only single function deployments are supported so the package is just the bytes of the
// python file
//
// 2. Create the function, with the reference to the previous package
//
// 3. Create the http trigger so the function can be accessed through the router
// TODO should check if function exists first
func createFunction(_ *cobra.Command, _ []string) error {

	// make fission client
	fissionClient, _, _, err := crd.MakeFissionClient()
	if err != nil {
		return errors.Wrap(err, "could not create fission client")
	}

	// first check if the function already exists
	_, err = fissionClient.CoreV1().Functions(DefaultNamespace).Get(fnName, metav1.GetOptions{})
	if err == nil {
		return errors.New(fmt.Sprintf("function \"%s\" already exists", fnName))
	}

	pkg, err := createPackage(fissionClient, fnName, fnCodePath)
	if err != nil {
		return err
	}
	fmt.Println("Created source code package")

	var secrets []fv1.SecretReference
	var cfgmaps []fv1.ConfigMapReference
	var resourceReq = v1.ResourceRequirements{}
	var idleTimeout = DefaultIdleTimeout

	fun := &fv1.Function{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fnName,
			Namespace: DefaultNamespace,
		},
		Spec: fv1.FunctionSpec{
			Environment: fv1.EnvironmentReference{
				Namespace: DefaultNamespace,
				Name:      DefaultEnvironment,
			},
			Package: fv1.FunctionPackageRef{
				FunctionName: "",
				PackageRef: fv1.PackageRef{
					Namespace:       pkg.Namespace,
					Name:            pkg.Name,
					ResourceVersion: pkg.ResourceVersion,
				},
			},
			InvokeStrategy: fv1.InvokeStrategy{
				ExecutionStrategy: fv1.ExecutionStrategy{
					SpecializationTimeout: fv1.DefaultSpecializationTimeOut,
					ExecutorType:          fv1.ExecutorTypePoolmgr,
				},
				StrategyType: fv1.StrategyTypeExecution,
			},
			Secrets:         secrets,
			ConfigMaps:      cfgmaps,
			Resources:       resourceReq,
			FunctionTimeout: DefaultTimeout,
			IdleTimeout:     &idleTimeout,
			Concurrency:     DefaultConcurrency,
		},
	}

	_, err = fissionClient.CoreV1().Functions(DefaultNamespace).Create(fun)
	if err != nil {
		return errors.Wrap(err, "could not create function")
	}

	// Create triggers with a certain method
	err = createTrigger(fissionClient, fnName, []string{http.MethodGet})
	if err != nil {
		return errors.Wrap(err, "error creating trigger for function")
	}

	fmt.Println("Function", fnName, "created")

	return nil

}

// createPackage returns
func createPackage(fissionClient *crd.FissionClient, fnName, codePath string) (*fv1.Package, error) {

	pkgName := fmt.Sprintf("%v-%v", fnName, "pkg")
	fmt.Println("Pkg name is ", pkgName)

	deployment, err := createArchive(codePath)
	if err != nil {
		return nil, errors.Wrap(err, "could not create archive from file")
	}

	pkg := &fv1.Package{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pkgName,
			Namespace: DefaultNamespace,
		},
		Spec: fv1.PackageSpec{
			Environment: fv1.EnvironmentReference{
				Namespace: DefaultNamespace,
				Name:      DefaultEnvironment,
			},
			Deployment: *deployment,
		},
		Status: fv1.PackageStatus{
			BuildStatus:         fv1.BuildStatusSucceeded,
			LastUpdateTimestamp: metav1.Time{Time: time.Now().UTC()},
		},
	}

	pkgRef, err := fissionClient.CoreV1().Packages(DefaultNamespace).Create(pkg)
	if err != nil {
		return nil, errors.Wrap(err, "could not create package")
	}

	return pkgRef, nil

}

// createArchive creates an archive from the code input by the user
// TODO for now only allow for single file literal deployments for convenience
func createArchive(codePath string) (*fv1.Archive, error) {

	absPath, err := filepath.Abs(codePath)
	if err != nil {
		return nil, errors.Wrap(err, "could not find code file")
	}
	fmt.Println("Using file ", absPath, "as the code of the function")

	var archive fv1.Archive
	size, err := fileSize(absPath)
	if err != nil {
		return nil, err
	}

	if size < fv1.ArchiveLiteralSizeLimit {
		archive.Type = fv1.ArchiveTypeLiteral
		archive.Literal, err = getFileContents(absPath)
		if err != nil {
			return nil, err
		}
	} else {
		return nil, errors.New("Only single file functions supported")
	}

	return &archive, nil

}

func getFileContents(path string) ([]byte, error) {
	code, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, errors.Wrapf(err, "error reading file %v", path)
	}

	return code, nil
}

func fileSize(path string) (int64, error) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}

// createTrigger creates an http route in fission with the same name of the
// function for convenience
func createTrigger(fissionClient *crd.FissionClient, name string, methods []string) error {

	for _, method := range methods {
		ht := &fv1.HTTPTrigger{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("%v-%v", name, strings.ToLower(method)),
				Namespace: DefaultNamespace,
			},
			Spec: fv1.HTTPTriggerSpec{
				RelativeURL: "/" + name,
				Method:      method,
				FunctionReference: fv1.FunctionReference{
					Type: fv1.FunctionReferenceTypeFunctionName,
					Name: name,
				},
				IngressConfig: fv1.IngressConfig{
					Annotations: nil,
					Path:        "/" + name,
					Host:        "*",
					TLS:         "",
				},
			},
		}

		_, err := fissionClient.CoreV1().HTTPTriggers(DefaultNamespace).Create(ht)
		if err != nil {
			return errors.Wrap(err, "unable to create http trigger")
		}

	}

	return nil

}

// deleteFunction deletes one of the functions
// TODO see if we can make the fission client be created on a prerun
func deleteFunction(_ *cobra.Command, _ []string) error {
	// make fission client
	fissionClient, _, _, err := crd.MakeFissionClient()
	if err != nil {
		return errors.Wrap(err, "could not initialize fission client")
	}

	// Should delete the function, the http triggers and the package
	var result *multierror.Error
	fmt.Println("Deleting function resource...")
	err = fissionClient.CoreV1().Functions(DefaultNamespace).Delete(fnName, &metav1.DeleteOptions{})
	result = multierror.Append(result, err)

	fmt.Println("Deleting package resource...")
	pkgName := fmt.Sprintf("%v-%v", fnName, "pkg")
	err = fissionClient.CoreV1().Packages(DefaultNamespace).Delete(pkgName, &metav1.DeleteOptions{})
	result = multierror.Append(result, err)

	fmt.Println("Deleting triggers...")
	triggerName := fmt.Sprintf("%s-%s", fnName, strings.ToLower(http.MethodGet))
	err = fissionClient.CoreV1().HTTPTriggers(DefaultNamespace).Delete(triggerName, &metav1.DeleteOptions{})
	result = multierror.Append(result, err)

	if err = result.ErrorOrNil(); err == nil {
		fmt.Printf("Function \"%s\" deleted", fnName)
		return nil
	}

	return result.ErrorOrNil()

}

// listFunctions returns a table with the information of the current functions
func listFunctions(_ *cobra.Command, _ []string) error {
	// make fission client
	fissionClient, _, _, err := crd.MakeFissionClient()
	if err != nil {
		return errors.Wrap(err, "Unable to create the fission client")
	}

	// get the list of functions and print some of their properties to a table
	funList, err := fissionClient.CoreV1().Functions("").List(metav1.ListOptions{})
	if err != nil {
		return errors.Wrap(err, "could not list functions")
	}

	w := tabwriter.NewWriter(os.Stdout, 1, 1, 2, ' ', 0)
	fmt.Fprintf(w, "%v\t%v\t%v\t%v\t%v\n", "NAME", "ENVIRONMENT", "CONCURRENCY", "TIMEOUT", "CREATED")

	// Display functions that use the default environment
	for _, fun := range funList.Items {
		if fun.Spec.Environment.Name == DefaultEnvironment {
			fmt.Fprintf(w, "%v\t%v\t%v\t%v\t%v\n", fun.Name, fun.Spec.Environment.Name, fun.Spec.Concurrency, fun.Spec.FunctionTimeout, fun.CreationTimestamp)
		}
	}

	w.Flush()

	return nil
}

func init() {
	rootCmd.AddCommand(functionCmd)
	functionCmd.AddCommand(functionCreateCmd)
	functionCmd.AddCommand(functionDeleteCmd)
	functionCmd.AddCommand(functionListCmd)

	// create command
	functionCreateCmd.Flags().StringVar(&fnName, "name", "", "Name of the function (required)")
	functionCreateCmd.Flags().StringVar(&fnCodePath, "code", "", "Path of the function file (required)")

	// delete command
	functionDeleteCmd.Flags().StringVar(&fnName, "name", "", "Name of the function (required)")

	// mark required fields
	functionCreateCmd.MarkFlagRequired("name")
	functionCreateCmd.MarkFlagRequired("code")
	functionDeleteCmd.MarkFlagRequired("name")

}
