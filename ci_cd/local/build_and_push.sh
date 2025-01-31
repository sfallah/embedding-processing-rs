#!/bin/bash
set -eu;

IMAGE_VERSION=$(git rev-parse --short=8 HEAD)
echo "Image version  $IMAGE_VERSION"

# Load environment variables from .env file (if exists)
if [ -f .env ]; then
  set -a  # Auto-export variables
  # shellcheck disable=SC1091
  source .env
  set +a
  echo "Loaded environment variables from .env file"
fi

echo "Using DockerHub user: $DOCKERHUB_USER"

docker login -u "$DOCKERHUB_USER" -p "$DOCKERHUB_PAT"

CUDA_DOCKER_ARCH="${CUDA_DOCKER_ARCH:-default}"

# Default to '12.2.2' if CUDA_VERSION is not set
CUDA_VERSION="${CUDA_VERSION:-12.2.2}"
echo "Using CUDA version: $CUDA_VERSION"

GITLAB_USER="${GITLAB_USER:-gitlab-ci-token}"
echo "Using Gitlab user: $GITLAB_USER"

GITLAB_TOKEN="${GITLAB_TOKEN:-$CI_JOB_TOKEN}"

if [ -z "$CUDA_VERSION" ]
  then

    echo "Compiling for CPU"
    REPO_NAME="qimia/embedding-server-rs"

    docker build --no-cache \
      --build-arg LLAMA_AVX512=OFF \
      --build-arg GITLAB_USER="$GITLAB_USER" \
      --build-arg GITLAB_TOKEN="$GITLAB_TOKEN" \
      --tag $REPO_NAME:"$IMAGE_VERSION" \
      -f .devops/Dockerfile .

else

    echo "Compiling for CUDA $CUDA_VERSION"
    REPO_NAME="qimia/embedding-server-rs-cuda"
    docker build --no-cache \
      --build-arg CUDA_VERSION="$CUDA_VERSION" \
      --build-arg GITLAB_USER="$GITLAB_USER" \
      --build-arg GITLAB_TOKEN="$GITLAB_TOKEN" \
      --build-arg CUDA_DOCKER_ARCH="$CUDA_DOCKER_ARCH" \
      --tag $REPO_NAME:"$IMAGE_VERSION" \
      -f .devops/cuda.Dockerfile .
fi

IMAGE_URI="$REPO_NAME:$IMAGE_VERSION"
echo "Pushing to $IMAGE_URI"
docker push "$IMAGE_URI"
