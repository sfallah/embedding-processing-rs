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

if [ -z "$2" ]; then
  CUDA_VERSION="12.2.2"
  echo "Using default CUDA version 12.2.2"
else
  CUDA_VERSION="$2"
  echo "Using CUDA version $CUDA_VERSION"
fi

GIT_COMMIT_ID=$(git rev-parse --short=8 HEAD)
IMAGE_VERSION="$IMAGE_BRANCH"-"$GIT_COMMIT_ID"

docker build --no-cache \
  --build-arg CUDA_VERSION="$CUDA_VERSION" \
  -t qimia/llama-zmq-server-base-cuda-"$CUDA_VERSION":"$IMAGE_VERSION" \
  -t qimia/llama-zmq-server-base-cuda-"$CUDA_VERSION":"$IMAGE_BRANCH"-latest \
  -f .devops/zmq-server-base-cuda.Dockerfile .

docker build --no-cache \
  --build-arg CUDA_VERSION="$CUDA_VERSION" \
  --build-arg LLAMA_BASE_VERSION="$IMAGE_VERSION" \
  --build-arg LLAMA_AVX512=OFF \
  -t qimia/llama-zmq-server-cuda-"$CUDA_VERSION":"$IMAGE_VERSION" \
  -t qimia/llama-zmq-server-cuda-"$CUDA_VERSION":"$IMAGE_BRANCH"-latest \
  -f .devops/llama-zmq-server-cuda.Dockerfile .

