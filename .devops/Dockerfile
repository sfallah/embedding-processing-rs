FROM lukemathwalker/cargo-chef:latest AS chef
WORKDIR /app

# lexrank-ndarray's blas-static feature builds OpenBLAS from source and links it into the
# binary, so the runtime image below needs no BLAS of its own.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential make \
    && rm -rf /var/lib/apt/lists/*

FROM chef AS planner
COPY ../Cargo.toml ../Cargo.lock ./
COPY ../.cargo ./.cargo
COPY ../src ./src
RUN cargo chef prepare

FROM chef AS builder
COPY --from=planner /app/recipe.json .
RUN cargo chef cook --release
COPY .. .
RUN cargo build --release
RUN mv ./target/release/embedding-server-rs ./app

FROM debian:stable-slim AS runtime
WORKDIR /app

# Splits already run in parallel through rayon; a threaded BLAS on top of that loses time on
# matrices this small (a chunk is at most a few dozen sentences).
ENV OPENBLAS_NUM_THREADS=1

COPY starter.sh .

CMD bash starter.sh