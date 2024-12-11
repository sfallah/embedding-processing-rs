#!/bin/bash
set -e

docker login

if [ -z "$1" ]; then
  IMAGE_BRANCH="dev"
  echo "Using default branch dev"
else
  IMAGE_BRANCH="$1"
  echo "Using branch $IMAGE_BRANCH"
fi

GIT_COMMIT_ID=$(git rev-parse --short=8 HEAD)
IMAGE_VERSION="$IMAGE_BRANCH"-"$GIT_COMMIT_ID"

docker build \
  -t qimia/llama-zmq-server-base:"$IMAGE_VERSION" \
  -t qimia/llama-zmq-server-base:"$IMAGE_BRANCH"-latest \
  -f .devops/zmq-server-base.Dockerfile .

docker push qimia/llama-zmq-server-base:"$IMAGE_VERSION"
docker push qimia/llama-zmq-server-base:"$IMAGE_BRANCH"-latest

docker build \
  --build-arg LLAMA_BASE_VERSION="$IMAGE_VERSION" \
  --build-arg LLAMA_AVX512=OFF \
  -t qimia/llama-zmq-server:"$IMAGE_VERSION" \
  -t qimia/llama-zmq-server:"$IMAGE_BRANCH"-latest \
  -f .devops/llama-zmq-server.Dockerfile .

docker push qimia/llama-zmq-server:"$IMAGE_VERSION"
docker push qimia/llama-zmq-server:"$IMAGE_BRANCH"-latest

