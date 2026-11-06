ARG BASE_IMAGE=public.ecr.aws/docker/library/ubuntu@sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates python3 python3-dev python3-pip python3-venv \
    gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu libc6-dev-arm64-cross libtinfo5 libtinfo-dev \
    zlib1g-dev libedit-dev libxml2-dev libgmp10 libatomic1 unzip file adb \
    pkg-config openssh-client && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/tilelang-venv
ENV PATH=/opt/tilelang-venv/bin:$PATH
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN pip install --no-cache-dir pip==25.2

# An optional local wheel avoids downloading PyTorch again. It is not in Git.
RUN --mount=type=bind,from=python-wheels,target=/wheels \
    if [ -f /wheels/torch-2.7.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl ]; then \
      pip install --no-cache-dir /wheels/torch-2.7.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl; \
    else \
      pip install --no-cache-dir torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu; \
    fi
COPY docker/hexagon/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

ENV USE_CUDA=OFF USE_ROCM=OFF USE_METAL=OFF CMAKE_POLICY_VERSION_MINIMUM=3.5
WORKDIR /opt/tilelang
COPY CMakeLists.txt VERSION ./
COPY cmake ./cmake
COPY src ./src
COPY 3rdparty ./3rdparty
COPY tilelang/jit/adapter/cython ./tilelang/jit/adapter/cython
COPY tilelang/jit/adapter/pto/cython_wrapper.pyx ./tilelang/jit/adapter/pto/cython_wrapper.pyx
COPY tilelang/ascend/torch_npu_exchange.pyx ./tilelang/ascend/torch_npu_exchange.pyx
COPY examples/ascend/deepgemm/_heuristics.pyx ./examples/ascend/deepgemm/_heuristics.pyx
ARG SOURCE_REVISION=unknown
ARG BUILD_JOBS=4
RUN printf '%s\n' "$SOURCE_REVISION" > /opt/tilelang/.git_commit.txt && \
    cmake -S /opt/tilelang -B /opt/tilelang/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DUSE_ROCM=OFF \
      -DUSE_METAL=OFF -DUSE_LLVM=OFF -DUSE_PYPI_Z3=ON && \
    cmake --build /opt/tilelang/build --parallel "$BUILD_JOBS"

# Docs, examples and launch scripts can change without recompiling C++.
COPY . /opt/tilelang

# Named local contexts are supplied by build-image.sh, never committed to Git.
COPY --from=hexagon-sdk --chown=1000:1000 . /opt/hexagon-sdk
COPY --from=android-ndk --chown=1000:1000 . /opt/android-ndk
COPY --from=board-runtime . /opt/board-runtime
RUN groupadd -g 1000 builder && \
    useradd -u 1000 -g 1000 -d /tmp/tilelang-home-1000 -M builder && \
    python /opt/tilelang/docker/hexagon/prepare-sdk.py && \
    mkdir -p /work /output /opt/tilelang/build/tvm && chmod 1777 /work /output
COPY docker/hexagon/tilelang-user.py /usr/local/bin/tilelang-entrypoint
RUN chmod 755 /usr/local/bin/tilelang-entrypoint

ENV PYTHONPATH=/opt/tilelang \
    HEXAGON_SDK_ROOT=/opt/hexagon-sdk \
    ANDROID_NDK_ROOT=/opt/android-ndk \
    CMAKE_ROOT_PATH=/opt/tilelang-venv \
    HEXAGON_DSP_ARCH=v73 \
    TILELANG_HEXAGON_HLOS=linux \
    TILELANG_HEXAGON_LINUX_RPC_LIB_DIR=/opt/board-runtime \
    TILELANG_HEXAGON_ELF_CRC=1 \
    TILELANG_HEXAGON_LEGACY_COMPUTE_RES=1 \
    ADB_SERVER_SOCKET=tcp:127.0.0.1:5037 \
    TMPDIR=/tmp \
    PYTHONDONTWRITEBYTECODE=1
LABEL org.opencontainers.image.title="TileLang Hexagon development environment" \
      org.opencontainers.image.revision=${SOURCE_REVISION}
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/tilelang-entrypoint"]
CMD ["bash"]
