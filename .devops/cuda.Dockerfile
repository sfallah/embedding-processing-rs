ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.2.2

ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}



FROM ${BASE_CUDA_DEV_CONTAINER} AS base-builder

ENV SCCACHE=0.5.4
ENV RUSTC_WRAPPER=/usr/local/bin/sccache
ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config libgflags-dev \
    clang \
    openssh-client \
    git \
    curl \
    cmake \
    ninja-build \
    libssl-dev \
    pkg-config \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Donwload and configure sccache
RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v$SCCACHE/sccache-v$SCCACHE-x86_64-unknown-linux-musl.tar.gz | tar -xzv --strip-components=1 -C /usr/local/bin sccache-v$SCCACHE-x86_64-unknown-linux-musl/sccache && \
    chmod +x /usr/local/bin/sccache

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN cargo install cargo-chef --locked

FROM base-builder AS build-deps

COPY gitlab.token /gitlab.token
COPY gitlab.username /gitlab.username

RUN GITLAB_TOKEN=$(cat /gitlab.token) && GITLAB_USERNAME=$(cat /gitlab.username) && \
    git config --global url."https://${GITLAB_USERNAME}:${GITLAB_TOKEN}@gitlab.com/".insteadOf "https://gitlab.com/"


# CUDA architecture to build for (defaults to all supported archs)
ARG CUDA_DOCKER_ARCH=default

# copy cuda libraries
ENV LD_LIBRARY_PATH=/usr/local/lib
COPY --from=base-builder /usr/local/cuda/lib64/libcublas.so ${LD_LIBRARY_PATH}/libcublas.so
COPY --from=base-builder /usr/local/cuda/lib64/libcublasLt.so ${LD_LIBRARY_PATH}/libcublasLt.so

RUN apt-get update && \
    apt-get install -y pkg-config libgflags-dev

RUN apt-get update && apt-get install -y libgtest-dev


RUN mkdir -p /usr/src/llama.cpp && \
    git clone --branch Release_b4153 https://gitlab.com/qimiaio/qimia-ai-dev/llama.cpp.git /usr/src/llama.cpp && \
    cd /usr/src/llama.cpp && \
    cmake -GNinja -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH} -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release && \
    cmake --install build --prefix /usr/local/llama

FROM build-deps AS planner

WORKDIR /usr/src/app

COPY ./.cargo ./.cargo
COPY ./embedding-common ./embedding-common
COPY ./embedding-database ./embedding-database
COPY ./embedding-index ./embedding-index
COPY ./embedding-processing ./embedding-processing
COPY ./embedding-server ./embedding-server
COPY Cargo.toml ./
COPY Cargo.lock ./

RUN cargo chef prepare  --recipe-path recipe.json

FROM build-deps AS builder

ARG GIT_SHA
ARG DOCKER_LABEL

# Limit parallelism
ARG RAYON_NUM_THREADS
ARG CARGO_BUILD_JOBS
ARG CARGO_BUILD_INCREMENTAL

# sccache specific variables
ARG ACTIONS_CACHE_URL
ARG ACTIONS_RUNTIME_TOKEN
ARG SCCACHE_GHA_ENABLED

WORKDIR /usr/src/app
COPY gitlab.token /gitlab.token
COPY gitlab.username /gitlab.username

RUN GITLAB_TOKEN=$(cat /gitlab.token) && GITLAB_USERNAME=$(cat /gitlab.username) && \
    git config --global url."https://${GITLAB_USERNAME}:${GITLAB_TOKEN}@gitlab.com/".insteadOf "https://gitlab.com/"


COPY --from=planner /usr/src/app/recipe.json recipe.json
COPY --from=build-deps /usr/local/llama/lib /usr/lib
COPY --from=build-deps /app/.devops/starter.sh /app/starter.sh

RUN cargo chef cook --release --no-default-features --recipe-path recipe.json && sccache -s

COPY ./.cargo ./.cargo
COPY ./embedding-common ./embedding-common
COPY ./embedding-database ./embedding-database
COPY ./embedding-index ./embedding-index
COPY ./embedding-processing ./embedding-processing
COPY ./embedding-server ./embedding-server
COPY Cargo.toml ./
COPY Cargo.lock ./
COPY config.toml ./

RUN cargo build --release --bin embedding-server --no-default-features && sccache -s;

FROM ${BASE_CUDA_DEV_CONTAINER}

RUN apt-get update && apt-get install -y wget libssl-dev libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*


WORKDIR /usr/src/app

COPY ./.devops/starter.sh .
COPY --from=builder /usr/src/app/target/release/embedding-server-rs /usr/local/bin/embedding-server-rs

EXPOSE 5556
CMD bash starter.sh
