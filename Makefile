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
# Clean target
clean:
	cargo clean

# Test target
test: build
	cargo test

test_processing: build
	cargo test -p embedding-processing -- --test-threads=1

bench_processing: release
	cargo bench -p embedding-processing


