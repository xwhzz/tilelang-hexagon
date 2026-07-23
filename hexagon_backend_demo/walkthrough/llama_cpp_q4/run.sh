#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG=${HEXAGON_DEMO_CONFIG:-$BUNDLE_ROOT/config.sh}
[[ -f "$CONFIG" ]] && source "$CONFIG"

TILELANG_ROOT=${TILELANG_ROOT:-$(cd "$BUNDLE_ROOT/.." && pwd)}
TILELANG_PYTHON=${TILELANG_PYTHON:-python3}
LLAMA_CPP_ROOT=${LLAMA_CPP_ROOT:-/path/to/llama.cpp}
LLAMA_HTP_BUILD=${LLAMA_HTP_BUILD:-$LLAMA_CPP_ROOT/build-snap/ggml/src/ggml-hexagon/htp-v79-prefix/src/htp-v79-build}
export PYTHONDONTWRITEBYTECODE=1

generate() {
    PYTHONPATH="$TILELANG_ROOT" "$TILELANG_PYTHON" \
        "$SCRIPT_DIR/step2_generate.py" --output-dir "$SCRIPT_DIR/generated"
}

trace() {
    generate
    printf '\n[1/5] TileLang PrimFuncs\n'
    rg -n '^def make_|Q4HMXIntrinEmitter|\.mma_atom\(|\.dequant_tile\(' \
        "$SCRIPT_DIR/step1_tilelang_kernel.py"

    printf '\n[2/5] Generated C ABI and hardware atoms\n'
    rg -n '^int32_t tl_walkthrough_|tl_hexagon_q4_0_dequant|tl_hexagon_hmx_mma_atom' \
        "$SCRIPT_DIR/generated/kernel_walkthrough_q4_m32_n256_k2048.cc"

    printf '\n[3/5] Manifest: symbols, layout, and resource owner\n'
    "$TILELANG_PYTHON" -m json.tool \
        "$SCRIPT_DIR/generated/kernel_walkthrough_q4_m32_n256_k2048.json"

    printf '\n[4/5] llama.cpp adapter: match, run, generated calls, registration\n'
    rg -n 'extern "C" int32_t tl_walkthrough_|static int matches|static int run|_kernel\(|constructor' \
        "$SCRIPT_DIR/step3_tl_ggml_adapter.cc"

    printf '\n[5/5] llama.cpp patch: CMake wiring and stock dispatch seam\n'
    rg -n 'TILELANG_WALKTHROUGH_SOURCES|tl_walkthrough_adapter|kernel_walkthrough|struct tl_op_ctx|tl_dispatch' \
        "$SCRIPT_DIR/step4_llama_cpp.patch"
}

verify() {
    generate
    local generated=$SCRIPT_DIR/generated/kernel_walkthrough_q4_m32_n256_k2048.cc
    local adapter=$SCRIPT_DIR/step3_tl_ggml_adapter.cc
    local symbol
    for symbol in \
        tl_walkthrough_q4_m32_n256_k2048_pack_kernel \
        tl_walkthrough_q4_m32_n256_k2048_dequant_kernel \
        tl_walkthrough_q4_m32_n256_k2048_compute_kernel; do
        rg -q "int32_t $symbol" "$generated"
        rg -q "$symbol" "$adapter"
    done
    rg -q 'tl_hexagon_q4_0_dequant_tile_32x32' "$generated"
    rg -q 'tl_hexagon_hmx_mma_atom' "$generated"
    rg -q 'if (tl_dispatch(&octx) == 0) return 0' \
        "$SCRIPT_DIR/step4_llama_cpp.patch" --fixed-strings
    printf 'walkthrough contracts: PASS\n'
    printf '  generated symbols match adapter declarations\n'
    printf '  generated Q4 dequant and HMX atoms are present\n'
    printf '  llama.cpp fallback seam is present\n'
    if [[ -f "$LLAMA_HTP_BUILD/compile_commands.json" ]]; then
        "$TILELANG_PYTHON" "$SCRIPT_DIR/verify_hexagon_compile.py" \
            --compile-db "$LLAMA_HTP_BUILD/compile_commands.json" \
            --walkthrough-dir "$SCRIPT_DIR"
    else
        printf '  Hexagon compile skipped: no configured compile_commands.json\n'
    fi
}

install_commands() {
    cat <<EOF
TL=$TILELANG_ROOT
DEMO=$SCRIPT_DIR
LCPP=$LLAMA_CPP_ROOT
HTP=\$LCPP/ggml/src/ggml-hexagon/htp

\$DEMO/run.sh generate
cp \$DEMO/generated/kernel_walkthrough_q4_m32_n256_k2048.cc \$HTP/
cp \$DEMO/step3_tl_ggml_adapter.cc \$HTP/tl_walkthrough_adapter.cc
git -C \$LCPP apply --check \$DEMO/step4_llama_cpp.patch
git -C \$LCPP apply \$DEMO/step4_llama_cpp.patch

cmake --preset arm64-android-snapdragon-release -S \$LCPP -B \$LCPP/build-walkthrough \\
  -DGGML_HEXAGON_TILELANG_WALKTHROUGH=ON \\
  -DGGML_HEXAGON_TILELANG_SOURCE_DIR=\$TL
cmake --build \$LCPP/build-walkthrough --target htp-v79 -j8
EOF
}

case "${1:-trace}" in
    generate) generate ;;
    trace) trace ;;
    verify) verify ;;
    install-commands) install_commands ;;
    *)
        echo "Usage: $0 {generate|trace|verify|install-commands}" >&2
        exit 2
        ;;
esac
