# Hexagon environment setup

This guide sets up the `hexagon-backend` branch on a Linux x86_64 development
host. Choose the Docker image or a local Python environment. Both compile
TileLang kernels on the host and execute them through FastRPC on a connected
Hexagon device.

The reference environment uses Ubuntu 22.04, Python 3.10, Hexagon SDK 6.6.0.0
with Hexagon Tools 19.0.07, and Android NDK r25c for Android targets. The
SA8650 Automotive Linux board uses the Linux ARM64 host compiler and the
FastRPC libraries supplied with its BSP (Board Support Package).

## Prerequisites

Obtain the following before choosing an installation method.

| Component | Requirement |
| --- | --- |
| Source | This repository's `hexagon-backend` branch and its pinned submodules |
| Hexagon SDK | Full Linux SDK, including `setup_sdk_env.source`, Hexagon Tools, QAIC, QuRT, and `utils/examples/dsp_capabilities_utils.{c,h}` |
| Linux target | SSH access, a working FastRPC installation, and a local copy of the target BSP's ARM64 runtime libraries |
| Android target | Android NDK r25c and an authorized ADB connection |
| Build host | Linux x86_64, network access for dependencies, and sufficient disk space for the SDK, source and build products |

The SDK's CI archive can omit the host example helpers needed by the FastRPC
build. Use an SDK distribution containing these files. Obtain the SDK and BSP
libraries through your Qualcomm or board-vendor distribution channel.

```bash
git clone --branch hexagon-backend --recurse-submodules \
  https://github.com/xwhzz/tilelang-hexagon.git
cd tilelang-hexagon
git submodule update --init --recursive
git rev-parse HEAD
```

Record the revision with acceptance results. Limit compiler parallelism to the
host's available memory, starting with two jobs on a small machine.

### Linux BSP runtime libraries

Create a local directory containing the target's `libcdsprpc.so` and its shared
library dependencies, including the matching C library and dynamic loader.
Preserve each dependency's filename and resolve symlinks when copying from a
BSP sysroot. The Linux host builder checks ARM64 architecture, dependency closure,
symbol versions and required dynamic symbols before deployment.

For a board that provides `ldd`, inspect its runtime after configuring SSH as
shown in [Linux board configuration](#linux-board-configuration).

```bash
ssh "$TILELANG_HEXAGON_SSH_HOST" 'ldd /usr/lib/libcdsprpc.so'
export BOARD_RUNTIME_PATH=/absolute/path/to/board-runtime
mkdir -p "$BOARD_RUNTIME_PATH"
scp -O "$TILELANG_HEXAGON_SSH_HOST:/usr/lib/libcdsprpc.so" "$BOARD_RUNTIME_PATH/"
```

Copy the resolved dependency files reported by `ldd` into the same directory.
Paths vary by BSP. If the build reports `Target runtime dependency not captured`,
copy the named library from that same BSP and retry. A symbol-version error
requires a host cross-toolchain/sysroot compatible with the target's C library.

## Option 1 — Docker

Use Docker Engine with BuildKit named build contexts, and a user account with
permission to run Docker. The image contains the checkout, native TileLang/TVM
libraries, Python dependencies, SDK, NDK and supplied BSP libraries. Its default
configuration targets the SA8650 Linux board.

### Build the image

Run from the repository root. This image includes both Linux and Android host
build tools, so the build takes both SDK and NDK directories.

```bash
export HEXAGON_SDK_PATH=/absolute/path/to/Hexagon_SDK/6.6.0.0
export ANDROID_NDK_PATH=/absolute/path/to/android-ndk-r25c
export BOARD_RUNTIME_PATH=/absolute/path/to/board-runtime
BUILD_JOBS=2 bash docker/hexagon/build-image.sh
```

| Build variable | Default or purpose |
| --- | --- |
| `IMAGE` | `tilelang-hexagon:sa8650-v73` |
| `BUILD_JOBS` | `4`, native compiler parallelism |
| `PIP_INDEX_URL` | `https://pypi.org/simple` |
| `BOARD_RUNTIME_PATH` | `docker/hexagon/local/board-runtime` |
| `PYTHON_WHEEL_DIR` | `docker/hexagon/local/wheelhouse`, optional CPU PyTorch wheel cache |

The optional wheel cache accepts
`torch-2.7.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl`. Otherwise the build
downloads it from PyTorch's CPU index. The remaining Python dependencies are
pinned in `docker/hexagon/requirements.lock`.

Keep toolchains and BSP libraries outside Git, for example under the ignored
`docker/hexagon/local/` directory. Follow their distribution terms when sharing
an image containing them.

### Enter the environment

Configure the target using the [device settings](#device-settings) below, then
run the launcher from a directory where you want to keep your own scripts and
outputs.

```bash
cd /path/to/my-work
bash /path/to/tilelang-hexagon/docker/hexagon/enter-docker.sh acceptance
```

This creates or starts `acceptance_hexagon` and opens a shell. The working
directory is mounted at `/work`, with `output/` also mounted at `/output`.
The repository built into the image is at `/opt/tilelang`. Files created in the
mounted directories use the caller's UID/GID. Leave the shell with `exit`.

For Linux SSH, the launcher mounts the caller's `~/.ssh` directory read-only
when present. Use `TILELANG_HEXAGON_SSH_DIR` to select another absolute directory. An
available `SSH_AUTH_SOCK` is forwarded for agent-based authentication. If
`TILELANG_HEXAGON_SSH_KNOWN_HOSTS` is set on the host, the launcher mounts that
file by its absolute host path and configures the container to use it. Paths inside a custom SSH config
must resolve inside the container.

For ADB, start and authorize the device on the host first. The container uses
host networking and the host ADB server at `127.0.0.1:5037`.

The launcher takes its image, environment and mounts when creating the
container. After changing those inputs, recreate your own container.

```bash
docker stop acceptance_hexagon
docker rm acceptance_hexagon
bash /path/to/tilelang-hexagon/docker/hexagon/enter-docker.sh acceptance
```

Rebuild the image to use an updated repository revision. For a direct shell
in an existing container, use its initialization entry point.

```bash
docker exec -it acceptance_hexagon /usr/local/bin/tilelang-entrypoint bash
```

## Option 2 — Local environment

The commands below use Ubuntu 22.04 and Python 3.10, matching the Docker
reference environment. Run them from the repository root.

### Install host dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential git ca-certificates python3 python3-dev python3-venv \
  gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu libc6-dev-arm64-cross \
  libncurses5 libtinfo5 libtinfo-dev zlib1g-dev libedit-dev libxml2-dev libgmp10 libatomic1 \
  unzip file pkg-config openssh-client adb

python3 -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.2
python -m pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r docker/hexagon/requirements.lock
```

The pinned `apache-tvm-ffi` version matches this checkout's native build.
Use the same Python environment for building and importing TileLang.

### Build TileLang

```bash
export CMAKE_POLICY_VERSION_MINIMUM=3.5
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_METAL=OFF \
  -DUSE_LLVM=OFF -DUSE_PYPI_Z3=ON
cmake --build build --parallel 2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_CACHE_DIR="$PWD/.cache/tilelang"
python -c 'import tilelang; print(tilelang.__file__)'
```

The import path should point into this checkout. Re-run
`cmake --build build --parallel 2` after changing native sources. Activate `.venv` and restore the
exports when opening a new shell. Each checkout should use its own build and
cache directories.

### Configure the SDK

```bash
export HEXAGON_SDK_ROOT=/absolute/path/to/Hexagon_SDK/6.6.0.0
export CMAKE_ROOT_PATH="$VIRTUAL_ENV"
```

`CMAKE_ROOT_PATH/bin` must contain both `cmake` and `ninja`, supplied here by
the Python packages. The runtime selects Hexagon Tools from the SDK and sources
`setup_sdk_env.source` in a build subprocess. Set the target variables in the
next section before validating the SDK.

For Android, also set the NDK location.

```bash
export ANDROID_NDK_ROOT=/absolute/path/to/android-ndk-r25c
```

The SDK CMake helpers expect an executable at `ipc/fastrpc/qaic/bin/qaic`.
The runtime creates a link to the SDK's prebuilt QAIC when necessary. With a
read-only SDK, have its installer prepare that path and execution permissions.
The Docker image prepares QAIC during the image build.

## Device settings

Select the target before compiling kernels. The architecture setting chooses
the generated DSP instructions. HLOS selects the board's host operating system,
and the FastRPC domain selects a DSP instance.

### Linux board configuration

For the SA8650 Automotive Linux board, set these variables on the host before
entering Docker, or in the activated local environment.

```bash
export TILELANG_HEXAGON_HLOS=linux
export HEXAGON_DSP_ARCH=v73
export TILELANG_HEXAGON_DOMAIN=3
export TILELANG_HEXAGON_SSH_HOST=user@board-hostname
export TILELANG_HEXAGON_REMOTE_DIR=/tmp/tilelang_acceptance
```

Use the board account permitted to access FastRPC. Configure noninteractive
SSH authentication and verify its host key against a trusted fingerprint.
Confirm the connection with the same strict settings used by the runtime.

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o "UserKnownHostsFile=${TILELANG_HEXAGON_SSH_KNOWN_HOSTS:-$HOME/.ssh/known_hosts}" \
  "$TILELANG_HEXAGON_SSH_HOST" true
```

SSH handles command execution, uploads, downloads and resident-agent tunnels.
The Linux runtime supports resident sessions for both compiled kernels and
multi-kernel programs. The board must allow loopback TCP forwarding for the
agent connection.

For a local environment, set the captured runtime directory as well.

```bash
export TILELANG_HEXAGON_LINUX_RPC_LIB_DIR=/absolute/path/to/board-runtime
```

The Docker image sets this to `/opt/board-runtime`. Local Linux builds use
`aarch64-linux-gnu-gcc` by default. `TILELANG_HEXAGON_LINUX_CC` selects another
compatible cross-compiler.

### SA8650 BSP compatibility

The SA8650 Automotive Linux BSP used for board validation requires two explicit
firmware compatibility settings. Select them according to the BSP and DSP
firmware, independently of `HEXAGON_DSP_ARCH`.

```bash
export TILELANG_HEXAGON_ELF_CRC=1
export TILELANG_HEXAGON_LEGACY_COMPUTE_RES=1
```

- `TILELANG_HEXAGON_ELF_CRC` adds the SWIV/CRC metadata required by the board's
  DSP ELF loader after linking each skeleton library. A missing section can
  produce `ELF verification: section header for CRC segment not found`.
  The implementation lives in [`elf_crc.py`](../tilelang/hexagon/runtime/elf_crc.py).
- `TILELANG_HEXAGON_LEGACY_COMPUTE_RES` supports firmware whose loader rejects
  the SDK's `compute_resource_attr_init_v2` import. The compatibility definition
  makes the SDK initializer fall back to `compute_resource_attr_init`.
  The definition lives in [`vtcm.h`](../src/tl_templates/hexagon/vtcm.h).

Both settings are opt-in in the Linux runtime and enabled by default in the
SA8650 Docker image and launcher. Set each to `0` when the target firmware
supports the corresponding standard loading or initialization path.

### Android configuration

For a v79 Android device, authorize it on the host and select its serial.

```bash
adb devices
export TILELANG_HEXAGON_HLOS=android
export HEXAGON_DSP_ARCH=v79
export TILELANG_HEXAGON_DEVICE=your-adb-serial
export TILELANG_HEXAGON_DOMAIN=3
export TILELANG_HEXAGON_REMOTE_DIR=/data/local/tmp/tilelang_acceptance
export TILELANG_HEXAGON_ELF_CRC=0
export TILELANG_HEXAGON_LEGACY_COMPUTE_RES=0
```

Use a separate container name when switching target configurations. A Linux
BSP with ADB can also use `TILELANG_HEXAGON_DEVICE` when
`TILELANG_HEXAGON_SSH_HOST` is unset. Select the serial explicitly on hosts with
several devices.

## Verify the environment

Run from the repository root in a local environment, or first run
`cd /opt/tilelang` inside Docker.

### Host checks

```bash
python -c 'import tilelang; print(tilelang.__file__)'
python -c 'from tilelang.hexagon.runtime.env import HexagonSDK; HexagonSDK().validate(); print("SDK paths OK")'
env -u TL_HEXAGON_DEVICE_TEST -u TL_HEXAGON_SIMULATOR_TEST \
  HEXAGON_DSP_ARCH=v79 \
  python -m pytest testing/python/hexagon -q
```

The host suite checks lowering and runtime orchestration against the reference
v79 target, including architecture-gated DMA cases. The architecture override
applies to this command only. Device and simulator cases are opt-in and appear
as skipped in this run. The board checks below use your selected device architecture.

### Simulator checks

The SDK simulator runs on the development host. The reference environment
includes its `libncurses.so.5` dependency. These opt-in tests cover conversions,
HVX math and Q4 dequantization across v73, v75 and v79.

```bash
export HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.07"
env -u TL_HEXAGON_DEVICE_TEST TL_HEXAGON_SIMULATOR_TEST=1 \
  python -m pytest \
  testing/python/hexagon/test_tilelang_hexagon_convert_sim.py \
  testing/python/hexagon/test_tilelang_hexagon_math_sim.py \
  testing/python/hexagon/test_tilelang_hexagon_dequant_sim.py -q
```

Device-only cases in these files are skipped by this command.

### Board smoke test

Reserve the device before running these commands. Start with a complete
64-by-64 tile, which uses contiguous DMA transfers.

```bash
python examples/hexagon/example_gemm.py \
  --m 64 --n 64 --k 64 --block 64 --block-k 64
TL_HEXAGON_DEVICE_TEST=1 python -m pytest \
  testing/python/hexagon/test_tilelang_hexagon_gemm.py \
  -k test_shapes_match_torch -q
```

Require `on HMX` and `(PASS)` from the example, then a passing pytest result.
The example prints its numerical result, so inspect that output as well as the
exit status. These checks exercise compilation, deployment, FastRPC, HMX and
the comparison with a CPU reference.

For acceptance, retain the source revision, SDK/Tools versions, BSP identity,
target settings and complete command output. Broader operator and model
coverage is described in the [example guide](../examples/hexagon/README.md).

On a shared device, coordinate with the other users and give each run its own
remote parent directory. `TILELANG_HEXAGON_DEVICE_LOCK` provides a host file lock
for cooperating processes. Across containers, all participants must mount the
same lock file or directory from the host. Remove only your own deployments
after their sessions have closed.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| CMake or Ninja missing | Activate the Python environment and check `$CMAKE_ROOT_PATH/bin` |
| QAIC fails or SDK helper is missing | Check the full SDK contents and QAIC execution permissions |
| Duplicate TVM FFI registrations on import | Use the pinned Python dependencies and this checkout's matching native build |
| SSH authentication or host-key failure | Test the strict SSH command above and verify the container's SSH mounts |
| Linux runtime dependency or ABI check fails | Capture the named dependency from the target BSP and use a compatible cross-toolchain |
| CRC section missing or `compute_resource_attr_init_v2` cannot resolve | Check the SA8650 BSP compatibility settings |
| An operator fails after the smoke test passes | Inspect that operator's generated code, architecture requirements and device log |
| Compiler runs out of memory | Reduce build parallelism and use a host with enough memory for the generated kernel |
