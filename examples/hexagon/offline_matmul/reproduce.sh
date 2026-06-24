#!/usr/bin/env bash
# Offline reproduce: generated Hexagon matmul kernel -> SDK build -> run on NPU -> compare golden.
#
# NO tilelang needed: the codegen (generate.py) is pre-done and the project C is
# committed.  Prerequisites (see README.md):
#   - Hexagon SDK build env on PATH (build_cmake, qaic, hexagon-clang++, NDK) — source your SDK env.
#   - python + numpy (for the golden inputs + the comparison).
#   - An authorized adb device (`adb devices`).
#
#   bash reproduce.sh            # arch defaults to v79; override: HEXAGON_DSP_ARCH=v73 bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
ARCH="${HEXAGON_DSP_ARCH:-v79}"
DEV=/data/local/tmp/tl_offline_mm
IFACE=tl_matmul_kernel

command -v build_cmake >/dev/null || { echo "build_cmake not on PATH — source your Hexagon SDK env first"; exit 1; }
command -v adb >/dev/null || { echo "adb not found"; exit 1; }
[ -f "project/${IFACE}_dsp.cc" ] || { echo "committed project missing — run 'python generate.py' once (needs tilelang)"; exit 1; }

echo "== 1/4  golden inputs (numpy only; the tilelang codegen is pre-done in ./project) =="
python golden.py

echo "== 2/4  build DSP skel + aarch64 host with the Hexagon SDK =="
# A host toolchain env (e.g. a conda env with compilers) must NOT leak into the
# Hexagon cross-compile — its CFLAGS (-march=nocona, -isystem .../include) make
# hexagon-clang fail.  Strip inherited compiler flags and start from a clean tree.
rm -rf project/hexagon_* project/android_*
( cd project
  unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS DEBUG_CFLAGS DEBUG_CXXFLAGS DEBUG_CPPFLAGS CMAKE_ARGS 2>/dev/null || true
  build_cmake hexagon DSP_ARCH="$ARCH"   # qaic + hexagon-clang++ -mhmx -mhvx -> lib${IFACE}_skel.so
  build_cmake android )                  # NDK aarch64 -> ${IFACE}_test
SKEL=$(find project -name "lib${IFACE}_skel.so" | head -1)
HOST=$(find project -name "${IFACE}_test" -type f | head -1)
[ -n "$SKEL" ] && [ -n "$HOST" ] || { echo "build artifacts not found"; exit 1; }
echo "   skel = $SKEL"
echo "   host = $HOST"

echo "== 3/4  push + run on the NPU (one-shot FastRPC host driver) =="
adb shell pkill -f _agent 2>/dev/null || true   # free VTCM from any resident agent
adb shell "mkdir -p $DEV"
adb push "$SKEL" "$HOST" project/A.bin project/B.bin "$DEV"/ >/dev/null
adb shell "cd $DEV && chmod 755 ${IFACE}_test && \
  LD_LIBRARY_PATH=. \
  ADSP_LIBRARY_PATH=$DEV:/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/vendor/dsp:/dsp \
  ./${IFACE}_test A.bin B.bin C.bin"          # argv = param order: A in, B in, C out
adb pull "$DEV/C.bin" project/C.bin >/dev/null

echo "== 4/4  compare device output vs golden reference =="
python verify.py
