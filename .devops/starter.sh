#!/bin/bash
set -eu
echo "Starting embedding server"
ls -lah /usr/src/app/target/release/
nvidia-smi
echo "llama path"
echo $LLAMA_PATH
echo "library path"
echo $LD_LIBRARY_PATH
ls -la ${LLAMA_PATH}/lib/
ls -la ${LLAMA_PATH}/include/
ls -la ${LLAMA_PATH}/bin/
# Only the API server. Insertion and query need the two model backends as well
# (target/release/model_cli and reranking_cli, on ports 5559 and 5557); this image has never
# started them, and the endpoints in config/config.toml point at localhost.
/usr/src/app/target/release/embedding_server --config-file /usr/src/app/config/config.toml --user-id "f1143e1a-828b-4e9b-8ac8-00c8cbd8cefe"
