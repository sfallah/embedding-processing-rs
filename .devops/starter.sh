#!/bin/bash
echo "Starting embedding server"
ls -la /usr/local/bin/embedding-server
nvidia-smi
echo "llama path"
echo $LLAMA_PATH
echo "llama path"
echo $LD_LIBRARY_PATH
ls -la ${LLAMA_PATH}/lib/
ls -la ${LLAMA_PATH}/include/
ls -la ${LLAMA_PATH}/bin/
/usr/local/bin/embedding-server --help
#--config-file /usr/src/app/config/config.toml --user-id "f1143e1a-828b-4e9b-8ac8-00c8cbd8cefe"
