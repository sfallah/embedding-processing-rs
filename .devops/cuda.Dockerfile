# Base Build Stage
ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.2.2
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}


FROM ${BASE_CUDA_DEV_CONTAINER} AS base-builder

ENV SCCACHE=0.5.4
ENV RUSTC_WRAPPER=/usr/local/bin/sccache
ENV PATH="/root/.cargo/bin:${PATH}"

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake clang libclang-dev libjemalloc-dev g++-12 gcc-12 \
    pkg-config libgflags-dev openssh-client git curl cmake ninja-build \
    libssl-dev python3 sse3-support \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && rm -rf /var/lib/apt/lists/*

# Install and configure sccache
RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v$SCCACHE/sccache-v$SCCACHE-x86_64-unknown-linux-musl.tar.gz \
    | tar -xzv --strip-components=1 -C /usr/local/bin sccache-v$SCCACHE-x86_64-unknown-linux-musl/sccache && \
    chmod +x /usr/local/bin/sccache

# Install Rust and cargo-chef
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    cargo install cargo-chef --locked

# Build Dependencies Stage
FROM base-builder AS build-deps

COPY gitlab.token /run/secrets/gitlab.token
COPY gitlab.username /run/secrets/gitlab.username

RUN git config --global url."https://$(cat /run/secrets/gitlab.username):$(cat /run/secrets/gitlab.token)@gitlab.com/".insteadOf "https://gitlab.com/"

ARG CUDA_DOCKER_ARCH=75
ARG LLAMA_CPP_VERSION=b4153

ENV LD_LIBRARY_PATH=/usr/local/lib
ENV LLAMA_CPP_BRANCH=Release_${LLAMA_CPP_VERSION}
ENV LLAMA_PATH=/usr/local/llama_${LLAMA_CPP_VERSION}

ENV CUDA_ARCH=${CUDA_DOCKER_ARCH}


# Additional dependencies for Llama build
RUN apt-get update && \
    apt-get install -y libgtest-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/src/llama.cpp && \
    git clone --branch ${LLAMA_CPP_BRANCH} https://gitlab.com/qimiaio/qimia-ai-dev/llama.cpp.git /usr/src/llama.cpp && \
    cd /usr/src/llama.cpp && \
    cmake -GNinja -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release && \
    cmake --install build --prefix ${LLAMA_PATH}

# Planner Stage
FROM build-deps AS planner

ARG LLAMA_CPP_VERSION=b4153
ENV LLAMA_PATH=/usr/local/llama_${LLAMA_CPP_VERSION}
ENV LD_LIBRARY_PATH=${LLAMA_PATH}/lib:${LD_LIBRARY_PATH}

COPY --from=build-deps ${LLAMA_PATH} ${LLAMA_PATH}



# Application Build Stage
WORKDIR /usr/src/app

COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM build-deps AS builder

WORKDIR /usr/src/app

ARG LLAMA_CPP_VERSION=b4153
ENV LLAMA_PATH=/usr/local/llama_${LLAMA_CPP_VERSION}
ENV LD_LIBRARY_PATH=${LLAMA_PATH}/lib:${LD_LIBRARY_PATH}

COPY --from=planner /usr/src/app/recipe.json recipe.json
COPY --from=build-deps ${LLAMA_PATH} ${LLAMA_PATH}

RUN cargo chef cook --release --no-default-features --recipe-path recipe.json

RUN cargo build --release --no-default-features

# Final Runtime Stage
FROM ${BASE_CUDA_DEV_CONTAINER} AS runtime

ARG LLAMA_CPP_VERSION=b4153
ENV LLAMA_PATH=/usr/local/llama_${LLAMA_CPP_VERSION}
ENV LD_LIBRARY_PATH=${LLAMA_PATH}/lib:${LD_LIBRARY_PATH}


WORKDIR /usr/src/app

COPY --from=build-deps ${LLAMA_PATH} ${LLAMA_PATH}

COPY ./.devops/starter.sh /usr/src/app/starter.sh
COPY --from=builder /usr/src/app/target/release/embedding-cli /usr/local/bin/embedding-cli

RUN chmod +x /usr/src/app/starter.sh

EXPOSE 5556
CMD /usr/src/app/starter.sh
