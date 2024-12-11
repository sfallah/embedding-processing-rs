```shell
sudo systemctl start docker
```

```shell
sudo docker build --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" \
--build-arg CUDA_COMPUTE_CAP=61 \
-t local/embedding-server-rs:latest -f .devops/cuda.Dockerfile .
```

```shell
sudo docker run -p 5556:5556 --rm --runtime=nvidia --gpus '"device=0"' local/embedding-server-rs:latest
```