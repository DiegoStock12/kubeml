package main

import (
	"encoding/json"
	"fmt"
	fv1 "github.com/fission/fission/pkg/apis/core/v1"
	"github.com/fission/fission/pkg/crd"
	"github.com/google/uuid"
	"github.com/pkg/errors"
	"io/ioutil"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"os"
	"path/filepath"
	"time"
)

const (
	DEFAULT_NAMESPACE = "default"
	DEFAULT_ENVIRONMENT = "torch"
)


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


// createPackage returns
func createPackage(fissionClient *crd.FissionClient, fnName, codePath string) (*fv1.Package, error) {

	pkgName := fmt.Sprintf("%v-%v", fnName, uuid.New().String()[:8])
	fmt.Println("Pkg name is ", pkgName)

	pkgSpec := fv1.PackageSpec{
		Environment:  fv1.EnvironmentReference{
			Namespace: DEFAULT_NAMESPACE,
			Name: DEFAULT_ENVIRONMENT,
		},
	}
	var pkgStatus fv1.BuildStatus = fv1.BuildStatusSucceeded


	deployment, err := createArchive(codePath)
	if err != nil {
		return nil, errors.Wrap(err, "could not create archive from file")
	}

	pkgSpec.Deployment = *deployment

	pkg := &fv1.Package{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pkgName,
			Namespace: DEFAULT_NAMESPACE,
		},
		Spec: pkgSpec,
		Status: fv1.PackageStatus{
			BuildStatus:         pkgStatus,
			LastUpdateTimestamp: metav1.Time{Time: time.Now().UTC()},
		},
	}

	pkgRef, err := fissionClient.CoreV1().Packages(DEFAULT_NAMESPACE).Create(pkg)
	if err != nil {
		return nil, errors.Wrap(err, "could not create package")
	}

	return pkgRef, nil

}

func main() {
	_ = os.Setenv("KUBECONFIG", "C:\\Users\\diego\\.kube\\config")


	fissionClient, _, _, err := crd.MakeFissionClient()
	if err != nil {
		panic(err)
	}

	fmt.Println("trying to create package")
	p, err := createPackage(fissionClient, "test", "./example.py")
	if err != nil {
		panic(err)
	}

	fmt.Println(p)
	fmt.Println(string(p.Spec.Deployment.Literal), p.ManagedFields)
	j, err := json.Marshal(p)
	fmt.Println(string(j))


}