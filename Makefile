# Variables
LLAMA_PATH=/usr/local/llama_b4570
DYLD_LIBRARY_PATH=$(LLAMA_PATH)/lib:$DYLD_LIBRARY_PATH
LD_LIBRARY_PATH=$(LLAMA_PATH)/lib:$LD_LIBRARY_PATH

export LLAMA_PATH
export LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH

# Default target
all: build

# Build target
build:
	cargo build

release:
	cargo build --release

release_metal: clean
	cargo build --release --features metal
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

cli: build
	@echo "file-path: $(FILE_PATH)"
	cargo run --package embedding-cli --bin embedding-cli -- index --file-path=$(FILE_PATH)

run_metal:
	cargo run -r -F metal --bin embedding-server -- --config-file config.toml

run_cuda:
	cargo run -r -F cuda --bin embedding-server -- --config-file config.toml


