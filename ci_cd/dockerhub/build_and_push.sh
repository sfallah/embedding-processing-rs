set -eu;
IMAGE_VERSION=$(cat .version) || true

mkdir -p ./build-artifacts/

docker login -u "$DOCKERHUB_USER" -p "$DOCKERHUB_PAT"


echo "Image version  $IMAGE_VERSION"


if [ -z "$CUDA_VERSION" ]
  then

    echo "Compiling for CPU"
    REPO_NAME="qimia/qimia-ai-embedding-server"

    docker build --no-cache \
      --build-arg LLAMA_AVX512=OFF \
      --build-arg CI_JOB_TOKEN="$CI_JOB_TOKEN" \
      --tag $REPO_NAME:$IMAGE_VERSION \
      -f .devops/Dockerfile .

else

    echo "Compiling for CUDA $CUDA_VERSION"
    REPO_NAME="qimia/qimia-ai-embedding-cuda"
    docker build --no-cache \
      --build-arg CUDA_VERSION="$CUDA_VERSION" \
      --build-arg CI_JOB_TOKEN="$CI_JOB_TOKEN" \
      --tag $REPO_NAME:$IMAGE_VERSION \
      -f .devops/cuda.Dockerfile .
fi

IMAGE_URI="$REPO_NAME:$IMAGE_VERSION"
echo "Pushing to $IMAGE_URI"
docker push "$IMAGE_URI"
