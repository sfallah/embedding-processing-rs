set -eu;
set -o allexport
source "ci_cd/aws/$ENV/ci_vars"
set +o allexport
IMAGE_VERSION=$(cat .version) || true

mkdir -p ./build-artifacts/

if [ -z "$IMAGE_VERSION" ]
  then
    IMAGE_VERSION=$(date +%s) # No image version was given, autogenerating it with the timestamp
    echo "Setting version from the timestamp $IMAGE_VERSION"
fi

echo "Image version  $IMAGE_VERSION"

echo "Login to AWS ECR"
# See https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/"
echo "Building the image"

if [ -z "$CUDA_VERSION" ]
  then
    REPO_NAME="qimia-ai-embedding-server-$ENV"

    docker build --no-cache \
      --tag $REPO_NAME:$IMAGE_VERSION \
      -f .devops/Dockerfile .

else

    REPO_NAME="qimia-ai-embedding-server-gpu-$ENV"
    docker build --no-cache \
      --build-arg CUDA_VERSION="$CUDA_VERSION" \
      --tag $REPO_NAME:$IMAGE_VERSION \
      -f .devops/cuda.Dockerfile .
fi

ECR_TAG="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$IMAGE_VERSION"
echo "Tagging the image $REPO_NAME:$IMAGE_VERSION with the version as $ECR_TAG"
docker tag "$REPO_NAME:$IMAGE_VERSION" "$ECR_TAG"

echo "Pushing"
docker push $ECR_TAG
echo $ECR_TAG > "./build-artifacts/ecr_tag"
echo $IMAGE_VERSION > "./build-artifacts/image_version"
