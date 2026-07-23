#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CONFIG=${HEXAGON_DEMO_CONFIG:-$BUNDLE_ROOT/config.sh}
[[ -f "$CONFIG" ]] && source "$CONFIG"

TILELANG_ROOT=${TILELANG_ROOT:-$(cd "$BUNDLE_ROOT/.." && pwd)}
TILELANG_PYTHON=${TILELANG_PYTHON:-python3}
LLAMA_CPP_ROOT=${LLAMA_CPP_ROOT:-/home/xwh/scratchpad-llamacpp}
LLAMA_HTP_BUILD=${LLAMA_HTP_BUILD:-$LLAMA_CPP_ROOT/build-snap/ggml/src/ggml-hexagon/htp-v79-prefix/src/htp-v79-build}
CMAKE_BIN=${CMAKE_BIN:-cmake}
DEVICE_DIR=${DEVICE_DIR:-/data/local/tmp/llamahtp}
MODEL_Q4=${MODEL_Q4:-LFM2-1.2B-Q4_0.gguf}
MODEL_Q8=${MODEL_Q8:-LFM2-1.2B-Q8_0.gguf}
DSP_ARCH=${DSP_ARCH:-v79}

ADB=(adb)
[[ -n "${DEVICE_SERIAL:-}" ]] && ADB+=( -s "$DEVICE_SERIAL" )

usage() {
    cat <<'EOF'
Usage: scripts/demo.sh COMMAND

Backend flow:
  check                 Check Python, SDK, adb, device, and llama.cpp paths
  lower                 Lower generic T.gemm to Hexagon C without a device
  test-codegen          Run offline Q4/Q8 layout and codegen contracts
  matmul                Run generic FP16 T.gemm on HMX
  workers               Run the 1-HMX/6-HVX worker-pool demo
  q4-standalone         Run explicit Q4/HMX atom correctness on-device
  q8-standalone         Run Q8/HVX dot atom correctness on-device

llama.cpp integration example:
  generate-q4           Regenerate all checked-in Q4 embedded kernels
  llama-walkthrough     Generate and trace one shape from TileLang to llama.cpp
  inspect-llama         Show generated symbols, CMake wiring, seam, and adapter
  build-q4              Rebuild the integrated v79 DSP skel
  deploy-q4             Deploy the bundled staged Q4 skel as the active skel
  bench-q4              Run LFM2 Q4 pp32 benchmark
  chat-q4               Run deterministic 8-token model generation
  status                Show active, stock, and TileLang skel hashes
  restore-stock         Restore the stock DSP skel

Presentation:
  serve                 Serve the offline slide site at http://127.0.0.1:8765
EOF
}

device_shell() {
    "${ADB[@]}" shell "cd '$DEVICE_DIR' && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp $*"
}

check() {
    printf '%-20s %s\n' "TileLang root" "$TILELANG_ROOT"
    printf '%-20s %s\n' "Python" "$TILELANG_PYTHON"
    printf '%-20s %s\n' "llama.cpp" "$LLAMA_CPP_ROOT"
    printf '%-20s %s\n' "HTP build" "$LLAMA_HTP_BUILD"
    printf '%-20s %s\n' "Hexagon SDK" "${HEXAGON_SDK_ROOT:-unset}"
    printf '%-20s %s\n' "Hexagon tools" "${HEXAGON_TOOLS_ROOT:-unset}"
    printf '%-20s %s\n' "Android NDK" "${ANDROID_NDK_ROOT:-unset}"
    "$TILELANG_PYTHON" -c 'import tilelang; print("tilelang import: OK")'
    command -v adb >/dev/null
    command -v "$CMAKE_BIN" >/dev/null || [[ -x "$CMAKE_BIN" ]]
    [[ -d "${HEXAGON_SDK_ROOT:-}" ]]
    [[ -d "${HEXAGON_TOOLS_ROOT:-}" ]]
    [[ -d "${ANDROID_NDK_ROOT:-}" ]]
    "${ADB[@]}" get-state
    "${ADB[@]}" shell "test -x '$DEVICE_DIR/llama-cli'"
    "${ADB[@]}" shell "test -f '$DEVICE_DIR/$MODEL_Q4'"
    printf 'demo environment: OK\n'
}

generate_q4() {
    local out=$BUNDLE_ROOT/results/generated
    mkdir -p "$out"
    for n in 128 512; do
        for k in 2048 8192; do
            PYTHONPATH="$TILELANG_ROOT" "$TILELANG_PYTHON" \
                "$TILELANG_ROOT/examples/hexagon/llama_cpp_integration/emit_embeddable.py" \
                --n "$n" --k "$k" --output-dir "$out"
        done
    done
    for k in 2048 8192; do
        PYTHONPATH="$TILELANG_ROOT" "$TILELANG_PYTHON" \
            "$TILELANG_ROOT/examples/hexagon/llama_cpp_integration/emit_staged_qmatmul.py" \
            --k "$k" --output-dir "$out"
    done
}

inspect_llama() {
    local htp=$LLAMA_CPP_ROOT/ggml/src/ggml-hexagon/htp
    [[ -d "$htp" ]]

    printf '\n[1/4] TileLang generated entry symbols\n'
    rg -n '^int32_t .*_kernel' "$htp"/kernel_qmatmul_hmx_staged_*.cc

    printf '\n[2/4] Generated translation units compiled into the DSP skel\n'
    rg -n 'tl_ggml_matmul|kernel_qmatmul_hmx_' "$htp/CMakeLists.txt"

    printf '\n[3/4] Stock hmx_mm_2d_f32 dispatch seam and fallback\n'
    rg -n -C 3 'struct tl_op_ctx octx|tl_dispatch\(&octx\)' "$htp/matmul-ops.c"

    printf '\n[4/4] Framework adapter: staged runner, match contract, registration\n'
    rg -n 'tl_q4_hmx_run_staged|tl_q4_hmx_matches|tl_q4_hmx_register' \
        "$htp/tl_ggml_matmul.cc"
}

case "${1:-help}" in
    check)
        check
        ;;
    lower)
        PYTHONPATH="$TILELANG_ROOT" "$TILELANG_PYTHON" "$SCRIPT_DIR/lower_example.py" \
            --tilelang-root "$TILELANG_ROOT" \
            --output "$BUNDLE_ROOT/results/generated_matmul_hexagon.c"
        ;;
    test-codegen)
        PYTHONPATH="$TILELANG_ROOT" "$TILELANG_PYTHON" "$SCRIPT_DIR/run_codegen_tests.py" \
            --tilelang-root "$TILELANG_ROOT"
        ;;
    matmul)
        PYTHONPATH="$TILELANG_ROOT" "$TILELANG_PYTHON" \
            "$TILELANG_ROOT/examples/hexagon/example_matmul.py" \
            --m 256 --n 256 --k 256 --block 64
        ;;
    workers)
        PYTHONPATH="$TILELANG_ROOT" "$TILELANG_PYTHON" \
            "$TILELANG_ROOT/examples/hexagon/example_worker_pool.py"
        ;;
    q4-standalone)
        PYTHONPATH="$TILELANG_ROOT" "$TILELANG_PYTHON" \
            "$TILELANG_ROOT/examples/hexagon/example_qmatmul_kstream.py" \
            --m 32 --n 128 --k 512 --io-dtype float32
        ;;
    q8-standalone)
        PYTHONPATH="$TILELANG_ROOT" "$TILELANG_PYTHON" \
            "$TILELANG_ROOT/examples/hexagon/example_qgemv_q8_0.py" --k 2048
        ;;
    generate-q4)
        generate_q4
        ;;
    llama-walkthrough)
        "$BUNDLE_ROOT/walkthrough/llama_cpp_q4/run.sh" trace
        ;;
    inspect-llama)
        inspect_llama
        ;;
    build-q4)
        "$CMAKE_BIN" --build "$LLAMA_HTP_BUILD" -j8
        ;;
    deploy-q4)
        "${ADB[@]}" push \
            "$BUNDLE_ROOT/artifacts/libggml-htp-${DSP_ARCH}.so.tilelang-q4-staged" \
            "$DEVICE_DIR/libggml-htp-${DSP_ARCH}.so"
        ;;
    bench-q4)
        device_shell "./llama-bench -m '$MODEL_Q4' -dev HTP0 -ngl 99 -p 32 -n 0 -r 3"
        ;;
    chat-q4)
        device_shell "./llama-cli -m '$MODEL_Q4' --device HTP0 -ngl 99 -n 1024 -st -p 'Introduce DENSO.' --temp 0 --seed 123"
        ;;
    status)
        "${ADB[@]}" shell "sha256sum '$DEVICE_DIR/libggml-htp-${DSP_ARCH}.so' '$DEVICE_DIR/libggml-htp-${DSP_ARCH}.so.stock' '$DEVICE_DIR/libggml-htp-${DSP_ARCH}.so.tilelang-q4-staged'"
        ;;
    restore-stock)
        if [[ -f "$BUNDLE_ROOT/artifacts/libggml-htp-${DSP_ARCH}.so.stock" ]]; then
            "${ADB[@]}" push \
                "$BUNDLE_ROOT/artifacts/libggml-htp-${DSP_ARCH}.so.stock" \
                "$DEVICE_DIR/libggml-htp-${DSP_ARCH}.so"
        else
            "${ADB[@]}" shell "cp '$DEVICE_DIR/libggml-htp-${DSP_ARCH}.so.stock' '$DEVICE_DIR/libggml-htp-${DSP_ARCH}.so'"
        fi
        ;;
    serve)
        "$TILELANG_PYTHON" -m http.server 8765 --directory "$BUNDLE_ROOT"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
