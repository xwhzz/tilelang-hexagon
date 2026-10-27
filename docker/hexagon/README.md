# Hexagon Docker development environment

Build TileLang and TVM from this checkout on a Linux x86_64 host. The resulting
image includes the source, native libraries, SDK, NDK and supplied board runtime;
running it does not require mounting a source checkout or a toolchain directory.
PyTorch runs on the x86 CPU; the Hexagon adapter builds and runs kernels on the
attached board. No CUDA installation or NVIDIA GPU is required.

This configuration targets an SA8650/v73 board with a Linux HLOS. Android/v79
settings are selectable but are not validated by this Docker contribution.

## Build

Use Docker Engine with BuildKit support for named local build contexts. The user
running the scripts must be allowed to use Docker. Initialize the pinned Git
submodules before building:

```bash
git submodule update --init --recursive
export HEXAGON_SDK_PATH=/path/to/Hexagon_SDK/6.6.0.0
export ANDROID_NDK_PATH=/path/to/android-ndk-r25c
export BOARD_RUNTIME_PATH=/path/to/board-runtime
bash docker/hexagon/build-image.sh
```

`BOARD_RUNTIME_PATH` is a local directory containing the target board's ARM64
`libcdsprpc.so` and its shared-library dependencies. The Linux host builder checks
their architecture, dependency closure and required symbol versions. Supply the
libraries from the same BSP as the target; Android host builds use the NDK instead.

The SDK, NDK, board libraries and optional wheel cache are build inputs, not Git
content. Keep them outside the checkout or under the ignored `docker/hexagon/local/`
directory. They are copied into the image at build time. Check their redistribution
terms before publishing an image containing them.

Supply an SDK containing `utils/examples/dsp_capabilities_utils.{c,h}`. The
official CI archive omits these host helpers; an unmodified CI archive is not a
complete input. A preinstalled `ipc/fastrpc/qaic/bin/qaic` is reused, otherwise
the SDK's QAIC Makefile installs it. Image preparation makes QAIC executable by
the runtime user, including when its original mode is 0750.

The default image is `tilelang-hexagon:sa8650-v73`. Optional build variables:

| Variable | Default / purpose |
| --- | --- |
| `IMAGE` | Image tag to build |
| `BUILD_JOBS` | `4`, limits native compiler parallelism |
| `PIP_INDEX_URL` | `https://pypi.org/simple`, Python package mirror |
| `PYTHON_WHEEL_DIR` | Optional local directory with the CPU torch 2.7.1 wheel |

An optional wheel cache must contain
`torch-2.7.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl` for this Ubuntu 22.04/Python
3.10 image. Without it, the build downloads the CPU wheel from PyTorch's index.
Other Python and build dependencies are pinned in `requirements.lock`.
Network access is needed for the Ubuntu base image, apt packages and Python
dependencies. A board is not needed to build the image. No prebuilt TileLang or
TVM library is copied from the development machine.

## Create or enter a container

Start the host ADB server and authorize/connect the board first. The container
uses host networking and the host ADB server at `127.0.0.1:5037`, so USB devices do
not need to be passed into Docker.

For a Linux board using SSH to run programs and ADB to transfer files:

```bash
export TILELANG_HEXAGON_DEVICE=192.168.1.1:5555
export TILELANG_HEXAGON_SSH_HOST=root@192.168.1.1
adb connect "$TILELANG_HEXAGON_DEVICE"
cd /path/to/my-work
bash /path/to/tilelang/docker/hexagon/enter-docker.sh myname
```

The example address and account above must be replaced with your board's values.
The name becomes `myname_hexagon`. The script creates it once, starts it if stopped,
and enters it through `tilelang-entrypoint` on every invocation. `exit` leaves the
interactive session; the container stays running in the background.

A network cable is sufficient when the board exposes TCP ADB and SSH. File
transfer still requires ADB, even when execution uses SSH. The Linux launcher
uses `/tmp/tilelang_<container>` as a remote parent writable by the ADB user;
override it with `TILELANG_HEXAGON_REMOTE_DIR`. Each kernel gets a unique child
directory for its libraries, executable and tensor files. Direct Python use
defaults to `/tmp/tilelang_hexagon`. These directories remain for inspection
after a run; remove them only after all kernels using them have closed.

`TILELANG_HEXAGON_DEVICE` selects the ADB serial on both Linux and Android. If
unset, the first authorized device is used. Set it explicitly when both a board
and a phone are connected. Existing Linux connections keep their transport
settings when environment variables change; compiling again uses a separate
cache entry for the new platform, architecture, toolchain or device settings.

The current terminal directory is mounted at `/work`, and its `output/` subdirectory
at `/output`. Files written there use the caller's UID/GID. The entrypoint also
registers that UID for OpenSSH and configures a writable home/cache directory.
On SSH entry, `connect-board` reads the board's public host key through ADB, stores
it in the container user's `known_hosts`, and keeps strict SSH host verification.
Override `TILELANG_HEXAGON_HOST_KEY_PATH` if the public key is not under
`/run/ssh/ssh_host_ed25519_key.pub`. No SSH private keys are included in the image.

Existing containers keep their original image, environment and mounts. To change
these, explicitly stop/remove the old container and run the script again:

```bash
docker stop myname_hexagon
docker rm myname_hexagon
```

For direct entry, preserve initialization:

```bash
docker exec -it myname_hexagon /usr/local/bin/tilelang-entrypoint bash
```

## Validate

Inside the container, run the repository's HMX example:

```bash
python /opt/tilelang/examples/hexagon/example_matmul.py --m 256 --n 256 --k 256 --block 64
```

An `on HMX: ... (PASS)` result confirms compilation, deployment, DSP execution
and the numerical check. Merely importing TileLang or generating C does not
validate board execution. The supplied SDK 6.6 QAIC binary is installed once when
building the image; SDK setup does not rebuild it on every kernel invocation.

The accompanying Linux runtime changes add ARM64 host compilation, optional SSH
execution, legacy ADB exit-status propagation and optional SA8650 ELF load CRC
metadata. `TILELANG_HEXAGON_LEGACY_COMPUTE_RES=1` explicitly enables a legacy
resource API fallback for BSPs whose loaders reject the SDK's optional v2
import. It is enabled in this SA8650 image and disabled by default outside the
image. Set it to `0` for Linux BSPs with the v2 API. The fallback is selected by
firmware compatibility, not by the DSP architecture. Android builds always
disable it, even if the environment still contains the Linux board setting.
`TILELANG_HEXAGON_ELF_CRC=1` separately enables the SA8650 load CRC metadata;
set it to `0` for BSPs that do not require it.

For an Android/v79 phone, create a new container with
`TILELANG_HEXAGON_HLOS=android`, `HEXAGON_DSP_ARCH=v79` and its explicit
`TILELANG_HEXAGON_DEVICE` serial. The launcher skips SSH initialization on this
path; the Android host and persistent agent continue to use ADB and the NDK.

Linux-specific orchestration and transport live in `tilelang/hexagon/runtime/linux_runtime.py`.
The original `device.py` is unchanged. `adapter.py` selects that connection only
for Linux, and `build.py` dispatches to its Linux build function. The architecture
priority is explicit argument, then `HEXAGON_DSP_ARCH`, then the existing v79 default.
The Linux build produces no persistent agent; the original agent selection logic
therefore uses the one-shot Linux driver. No diagnostic log-file or remote timeout
environment switches are added.

Validation on 2026-10-21, before this runtime-isolation refactor, used upstream
revision `582171a8a3f2d0b68c4e716fc43fe813bd5d4393` plus the Linux/v73 patches:

- A fresh source build succeeded with the ARM64 libc development headers installed.
- The host-only pytest run reported 25 passed and 5 skipped.
- HMX matmul 256x256x256 (block 64) passed without host directory mounts.
- The launcher, caller UID/GID mapping, and bind-mount writes passed.
- A user-provided FP32 Add 1024x1024 passed on the board through the launcher.

After the runtime-isolation refactor, the container-based host regression
tests passed. The image was rebuilt, and HMX matmul 256x256x256 (block 64)
passed again on SA8650/v73 without host directory mounts.
The Add result above predates the refactor. Android/v79 hardware and its
performance have not been validated by this contribution.

Validation on 2026-10-22 after the deployment/cache/SDK fixes:

- 42 host regression tests passed, including platform/device cache separation,
  same-named kernel deployment isolation and frozen Linux transport settings.
- The full SDK's QAIC was reinstalled as root during image preparation and then
  executed successfully by UID 1000; no manual permission fix was needed.
- Fresh DSL compilation of the original matmul example at sizes 256 and 128
  (block 64) produced two distinct deployments under the default `/tmp` parent.
  Five alternating calls passed finite-value and FP32-golden checks with maximum
  absolute error 0.0016332, below 0.01. Disassembly contained HMX and no DMA
  instructions. Changing SSH/ADB environment settings did not redirect the
  existing kernels.
- SDK 6.6 / Tools 19.0.07 resource-initialization probes for v73 and v79 matched
  the pre-contribution machine code when legacy compatibility was disabled.
- Android/v79 DSP, ARM host and persistent agent builds passed. An Android/v73
  DSP build also preserved the optional v2 import while the Linux compatibility
  environment variable was set, confirming the Android build disables it.

This run used an incremental image over the previously validated source-built
TileLang/TVM image: changed Python, DSP headers and Docker scripts were replaced;
the native compiler was reused. Each ARM host and DSP kernel was freshly built.
These are correctness/build checks, not performance measurements or Android
phone execution results.
