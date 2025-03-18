# BUILD

## Set up the environment
```shell
export LLAMA_PATH=/usr/local/llama_b4570
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$LLAMA_PATH/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LLAMA_PATH/lib
```

## Build
```shell
cargo build -r
```

## Run
```shell
cargo run -r --bin embedding-server -- --user-id "f1143e1a-828b-4e9b-8ac8-00c8cbd8cefe" 
```
## Test
```shell
cargo test
```