# Default target
all: build

# Build target
build:
	cargo build

release:
	cargo build --release

# `cuda` and `metal` exist only on the two model crates; they forward to qllama, whose build
# script compiles vendored llama.cpp from source, so the first build of these is slow.
release_metal:
	cargo build --release -F metal -p embedding-model -p reranking-model

release_cuda:
	cargo build --release -F cuda -p embedding-model -p reranking-model

# Clean target
clean:
	cargo clean

# Test target
test: build
	cargo test -- --test-threads=1

test_processing: build
	cargo test -p embedding-processing -- --test-threads=1

bench_processing: release
	cargo bench -p embedding-processing

bench_reranking_metal:
	cargo bench -F metal -p reranking-model

bench_reranking_cuda:
	cargo bench -F cuda -p reranking-model

# Three process groups, three terminals: the two model backends first, then the server. The
# server does not link llama.cpp, so it has no metal or cuda variant.
run_model_metal:
	cargo run -r -F metal -p embedding-model --bin model_cli -- --config-file embedding-model/embedding_config.toml

run_reranking_metal:
	cargo run -r -F metal -p reranking-model --bin reranking_cli -- --config-file reranking-model/reranking_config.toml

run_server:
	cargo run -r --bin embedding_server -- --config-file config.toml

# Kept: `make run_metal` is in the older docs and in muscle memory.
run_metal: run_server

# End-to-end check against a running server and both backends.
smoke:
	cargo run -r -p embedding-server --bin smoke_client

# Read a store written by the previous storage layer into per-workspace shards.
migrate:
	cargo run -r -p embedding-store -F rocksdb-migration --bin migrate_rocksdb -- \
		--from rocksdb_dir --config-file config.toml
