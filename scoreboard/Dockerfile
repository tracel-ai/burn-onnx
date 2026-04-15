FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install Python, git, and build essentials
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    locales && \
    apt-get clean autoclean && apt-get autoremove -y

RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

# Install Rust toolchain to a shared location so both the root build phase
# and the non-root runner user can use it. Ownership is transferred to the
# runner user later so the toolchain isn't world-writable.
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable --no-modify-path

# Copy scoreboard test infrastructure
COPY ./test /root/test
COPY ./setup /root/setup

# Install Python test dependencies
RUN pip3 install --no-cache-dir -r /root/setup/requirements_report.txt

############## ONNX Backend dependencies ###########
ENV ONNX_BACKEND="burn_onnx_backend"
ENV PYTHONPATH="/root"

# Install onnx
RUN pip3 install onnx numpy

# Clone and build burn-onnx (onnx2burn CLI)
ARG BURN_ONNX_REV=main
RUN git clone https://github.com/tracel-ai/burn-onnx.git /opt/burn-onnx && \
    cd /opt/burn-onnx && \
    git checkout ${BURN_ONNX_REV} && \
    cargo build --release -p burn-onnx --bin onnx2burn && \
    cp target/release/onnx2burn /usr/local/bin/

# Copy and pre-build the runner template (compiles burn deps once).
# Debug profile is intentional: it matches the per-model build in backend.py,
# so incremental rebuilds reuse the cached dep artifacts. Release would
# double the compile work without benefiting the short-lived test inferences.
COPY ./backends/burn-onnx/runner /root/runner
COPY ./backends/burn-onnx/backend.py /root/burn_onnx_backend.py
RUN cd /root/runner && cargo build

ENV BURN_RUNNER_DIR="/root/runner"
ENV ONNX2BURN="/usr/local/bin/onnx2burn"
####################################################

# Create runner user and transfer ownership of the Rust toolchain and cargo
# cache so incremental builds from backend.py can write to them, without
# making them world-writable. /root is also chowned so the pre-built runner
# target/ directory is usable.
RUN useradd -m runner && \
    chown -R runner:runner /root /opt/rustup /opt/cargo
USER runner
WORKDIR /root

CMD ["sh", "-c", ". /root/setup/docker-setup.sh && pytest /root/test/test_backend.py --onnx_backend=${ONNX_BACKEND} -k 'not _cuda' -v"]
