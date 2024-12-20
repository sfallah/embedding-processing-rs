#!/bin/bash
set -e

#docker login

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

if [ -z "$3" ]; then
  CUDA_DOCKER_ARCH="61"
else
  CUDA_DOCKER_ARCH="$3"
fi
echo "Using CUDA_DOCKER_ARCH = $CUDA_DOCKER_ARCH"


GIT_COMMIT_ID=$(git rev-parse --short=8 HEAD)
IMAGE_VERSION="$IMAGE_BRANCH"-"$GIT_COMMIT_ID"

docker build \
  --build-arg CUDA_VERSION="$CUDA_VERSION" \
  --build-arg CUDA_DOCKER_ARCH="$CUDA_DOCKER_ARCH" \
  -t qimia/qimia-ai-embedding-cuda-"$CUDA_VERSION":"$IMAGE_VERSION" \
  -t qimia/qimia-ai-embedding-cuda-"$CUDA_VERSION":"$IMAGE_BRANCH"-latest \
  -f .devops/cuda.Dockerfile .

