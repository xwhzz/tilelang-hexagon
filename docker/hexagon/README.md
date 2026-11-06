# Hexagon Docker environment

Build and run the `hexagon-backend` checkout in a Linux x86_64 container.
The image includes TileLang/TVM, the Hexagon SDK, Android NDK and supplied
Linux BSP libraries. Its defaults target the SA8650 Automotive Linux board.

See the [environment setup guide](../../docs/hexagon_env_setup.md) for inputs,
build and launch commands, SSH/ADB configuration, SA8650 firmware settings,
and verification.

| File | Purpose |
| --- | --- |
| `Dockerfile` | Build the native libraries and package the toolchains and runtime |
| `Dockerfile.dockerignore` | Exclude local environments, caches and build products |
| `requirements.lock` | Pin the reference Python environment |
| `build-image.sh` | Pass local SDK, NDK, BSP runtime and optional wheel contexts to Docker |
| `enter-docker.sh` | Create or enter a named container with work and SSH mounts |
| `prepare-sdk.py` | Prepare QAIC for the container's runtime user |
| `tilelang-user.py` | Initialize the caller's UID/GID and writable home, then run the command |

Local SDKs, board libraries and wheel caches can be kept under the ignored
`local/` directory.
