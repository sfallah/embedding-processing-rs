set -e
set -o allexport
source "ci_cd/aws/$ENV/ci_vars"
set +o allexport
if [ -z "$CUDA_VERSION" ]
  then
    REPO_NAME="qimia-ai-embedding-server-$ENV"
else
    REPO_NAME="qimia-ai-embedding-server-gpu-$ENV"
fi
ECR_TAG=$(cat "./build-artifacts/ecr_tag")
IMAGE_VERSION=$(cat "./build-artifacts/image_version")
CLUSTER_NAME="qimia-ai-$ENV"
SERVICE_NAME="$CLUSTER_NAME-ec2"

MANIFEST=$(aws ecr batch-get-image --repository-name "$REPO_NAME" --image-ids "imageTag=$IMAGE_VERSION" --output json | jq --raw-output '.images[0].imageManifest')
aws ecr put-image --repository-name "$REPO_NAME" --image-tag latest --image-manifest "$MANIFEST" --no-cli-pager
aws ecs update-service --cluster "$CLUSTER_NAME" --service "$SERVICE_NAME" --force-new-deployment  --no-cli-pager
