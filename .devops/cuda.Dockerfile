# Base Build Stage
ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.2.2
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}


FROM ${BASE_CUDA_DEV_CONTAINER} AS base-builder

#ENV SCCACHE=0.5.4
#ENV RUSTC_WRAPPER=/usr/local/bin/sccache
ENV PATH="/root/.cargo/bin:${PATH}"

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake clang libclang-dev libjemalloc-dev g++-12 gcc-12 \
    pkg-config libgflags-dev openssh-client git curl cmake ninja-build \
    libssl-dev python3 sse3-support libgtest-dev \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust and cargo-chef
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

ARG GITLAB_USER
ARG GITLAB_TOKEN

RUN git config --global url."https://${GITLAB_USER}:${GITLAB_TOKEN}@gitlab.com/".insteadOf "https://gitlab.com/"

ARG CUDA_DOCKER_ARCH=default
ARG LLAMA_CPP_VERSION=b4570

# Building llama.cpp here is very likely dead weight: the `llama-cpp` crate vendors llama.cpp and
# compiles it from source itself, and its build script does not read LLAMA_PATH or LD_LIBRARY_PATH.
# Left in place because no CUDA machine was available to verify the image without it. Removing this
# stage would roughly halve the image build; check first that the crate's build script gets
# -DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH} some other way.

ENV LLAMA_CPP_BRANCH=release_${LLAMA_CPP_VERSION}
ENV LLAMA_PATH=/usr/local/llama_${LLAMA_CPP_VERSION}
ENV LD_LIBRARY_PATH=${LLAMA_PATH}/lib:${LD_LIBRARY_PATH}


RUN if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
            export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
            fi && \
    mkdir -p /usr/src/llama.cpp && \
    git clone --branch ${LLAMA_CPP_BRANCH} https://gitlab.com/qimiaio/qimia-ai-dev/llama.cpp.git /usr/src/llama.cpp && \
    cd /usr/src/llama.cpp && \
    cmake -GNinja -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . &&\
    cmake --build build --config Release && \
    cmake --install build --prefix ${LLAMA_PATH}


# Final Runtime Stage
FROM base-builder AS runtime

WORKDIR /usr/src/app


COPY . .


WORKDIR /usr/src/app

COPY --from=base-builder ${LLAMA_PATH} ${LLAMA_PATH}

# Three process groups, three sets of binaries. Only the two model crates take `cuda`; the server
# does not link llama.cpp.
RUN cargo build --release --bin embedding_server \
 && cargo build --release -F cuda -p embedding-model -p reranking-model

COPY ./.devops/starter.sh /usr/src/app/starter.sh

# Splits already run in parallel through rayon; a threaded BLAS on top of that loses time on
# matrices this small (a chunk is at most a few dozen sentences). The server's OpenBLAS comes
# from lexrank-ndarray's blas-static feature and is linked into the binary.
ENV OPENBLAS_NUM_THREADS=1

RUN chmod +x /usr/src/app/starter.sh

EXPOSE 5556
CMD /usr/src/app/starter.sh
