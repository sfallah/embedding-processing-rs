# embedding-processing-rs

A RAG-style embedding service: documents are split, embedded with a GGUF model through llama.cpp,
summarised with LexRank, stored in a per-workspace record log, indexed in usearch (HNSW), and
queried — optionally reranked — over ZeroMQ with MessagePack payloads.

It runs as **three independent process groups**, each with its own config file and ports:

| Process group | Crate | Binary | Frontend port |
|---|---|---|---|
| API server | `embedding-server` | `embedding_server` | 5556 |
| Embedding model backend | `embedding-model` | `model_cli` | 5559 |
| Reranking model backend | `reranking-model` | `reranking_cli` | 5557 |

## Build

```shell
cargo build -r                                                        # server and libraries
cargo build -r -F metal -p embedding-model -p reranking-model         # Apple GPU
cargo build -r -F cuda  -p embedding-model -p reranking-model         # NVIDIA
```

`cuda` and `metal` exist only on the two model crates; the server does not link llama.cpp. The
`qllama` crate builds vendored llama.cpp from source with CMake, so a C/C++ toolchain and
`cmake` are required and the first build is slow. Its output is thousands of lines: send it to a
file rather than to the terminal.

Put the GGUF files in `models/` and point `config.toml` and `embedding-model/embedding_config.toml`
at the same one.

## Run

Start the two model backends first, then the server — each in its own terminal.

```shell
make run_model_metal        # or: cargo run -r -F metal -p embedding-model --bin model_cli -- ...
make run_reranking_metal
make run_server
```

Every binary takes `--config-file <toml>` and `--log-level trace|debug|info|warn|error`.

## Test

```shell
make test                   # cargo test -- --test-threads=1
make smoke                  # end-to-end against a running server and both backends
```

`embedding-processing`'s integration tests and both `benches/` suites talk to live model backends,
so start those first; the other crates' tests are self-contained.

## Migrating from the previous storage layer

A store written by the RocksDB layer is read into per-workspace shards by:

```shell
make migrate                # cargo run -r -p embedding-store -F rocksdb-migration --bin migrate_rocksdb -- ...
```

It leaves the source untouched, writes one shard per workspace, and exits non-zero if any document
could not be migrated, having migrated the rest.

## More

Deployment is in `.devops/DOCKER.md`.
