#!/usr/bin/env bash
# Usage: bash enter-docker.sh <name>
set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: bash $0 <name>"
  exit 1
fi

CONTAINER_NAME="${1}_hexagon"
IMAGE=${IMAGE:-tilelang-hexagon:sa8650-v73}
HLOS=${TILELANG_HEXAGON_HLOS:-linux}
if [ "$HLOS" = linux ]; then
  REMOTE_PARENT=/tmp/tilelang_${CONTAINER_NAME}
else
  REMOTE_PARENT=/data/local/tmp/tilelang_${CONTAINER_NAME}
fi
WORK_DIR="$PWD"
OUTPUT_DIR="$WORK_DIR/output"
mkdir -p "$OUTPUT_DIR"

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  docker start "$CONTAINER_NAME" >/dev/null
else
  SSH_ARGS=()
  if [ "$HLOS" = linux ] && [ -n "${TILELANG_HEXAGON_SSH_HOST:-}" ]; then
    SSH_DIR=${TILELANG_HEXAGON_SSH_DIR:-$HOME/.ssh}
    CONTAINER_HOME=/tmp/tilelang-home-$(id -u)
    if [ "$(id -u)" = 0 ]; then
      CONTAINER_HOME=/root
    fi
    if [ -d "$SSH_DIR" ]; then
      SSH_ARGS+=(-v "$SSH_DIR:$CONTAINER_HOME/.ssh:ro")
    fi
    if [ -n "${TILELANG_HEXAGON_SSH_KNOWN_HOSTS:-}" ]; then
      if [ ! -f "$TILELANG_HEXAGON_SSH_KNOWN_HOSTS" ]; then
        echo "SSH known_hosts file missing: $TILELANG_HEXAGON_SSH_KNOWN_HOSTS" >&2
        exit 1
      fi
      SSH_ARGS+=(-v "$TILELANG_HEXAGON_SSH_KNOWN_HOSTS:/run/tilelang-known-hosts:ro")
      SSH_ARGS+=(-e TILELANG_HEXAGON_SSH_KNOWN_HOSTS=/run/tilelang-known-hosts)
    fi
    if [ -n "${SSH_AUTH_SOCK:-}" ] && [ -S "$SSH_AUTH_SOCK" ]; then
      SSH_ARGS+=(-v "$SSH_AUTH_SOCK:/run/tilelang-ssh-agent")
      SSH_ARGS+=(-e SSH_AUTH_SOCK=/run/tilelang-ssh-agent)
    fi
  fi
  docker run -d --name "$CONTAINER_NAME" --network=host \
    -e LOCAL_UID="$(id -u)" -e LOCAL_GID="$(id -g)" \
    -e TMPDIR=/tmp \
    -e HEXAGON_DSP_ARCH="${HEXAGON_DSP_ARCH:-v73}" \
    -e TILELANG_HEXAGON_HLOS="$HLOS" \
    -e TILELANG_HEXAGON_LEGACY_COMPUTE_RES="${TILELANG_HEXAGON_LEGACY_COMPUTE_RES:-1}" \
    -e TILELANG_HEXAGON_ELF_CRC="${TILELANG_HEXAGON_ELF_CRC:-1}" \
    -e TILELANG_HEXAGON_DOMAIN="${TILELANG_HEXAGON_DOMAIN:-3}" \
    -e TILELANG_HEXAGON_DSP_LIBRARY_PATH \
    -e TILELANG_HEXAGON_DEVICE \
    -e TILELANG_HEXAGON_SSH_HOST \
    -e TILELANG_HEXAGON_SSH_TRANSFER_TIMEOUT \
    -e TILELANG_HEXAGON_REMOTE_DIR="${TILELANG_HEXAGON_REMOTE_DIR:-$REMOTE_PARENT}" \
    -v "$WORK_DIR:/work" -v "$OUTPUT_DIR:/output" \
    "${SSH_ARGS[@]}" \
    "$IMAGE" sleep infinity >/dev/null
fi

exec docker exec -it "$CONTAINER_NAME" \
  /usr/local/bin/tilelang-entrypoint bash
