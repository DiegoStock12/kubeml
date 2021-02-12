# 1) Download dependencies first so they are cached
FROM golang:1.12-alpine AS godep
RUN apk add bash ca-certificates git gcc g++ libc-dev

ENV GO111MODULE=on

ARG GOPKG=github.com/diegostock12/kubeml/ml

WORKDIR /app

# To reuse build cache, copy go.mod & go.sum and download dependencies first.
COPY go.mod .
COPY go.sum .
RUN go list ./...



# 2) Now copy the actual go files and build the application
FROM godep as builder

ARG GOPKG
WORKDIR /app

# Copy whole ml directory to work dir
COPY ./ ./

# Change working directory to where the stuff is
WORKDIR /app/cmd/ml


# Build the application
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -o kubeml



# 3) Finally use another container to slim down the image size
FROM alpine:3.10 as base
RUN apk add --update ca-certificates
COPY --from=builder /app/cmd/ml/kubeml /

ENTRYPOINT ["/kubeml"]