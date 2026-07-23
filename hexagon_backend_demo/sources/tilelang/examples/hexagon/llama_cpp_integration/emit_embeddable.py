"""Emit a llama.cpp-embeddable Q4_0 matmul built from explicit HMX atoms.

The generated kernel consumes ggml-hexagon's native 576-byte Q4_0 tiles.  It
does not require the legacy qcm/scale repack cache, a row-major FP16 weight
matrix, ``T.gemm``, an opaque dequant helper, or the monolithic HMX GEMM helper.

The ABI intentionally uses FP32 activations and outputs because that is what
``hmx_mm_2d_f32`` exposes in ggml-hexagon.  Conversion to and from HMX FP16
Crouton tiles happens inside the TileLang kernel.

Example:

    python emit_embeddable.py --n 128 --k 2048
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tilelang
from tilelang.utils.target import determine_target

from examples.hexagon.example_qmatmul_kstream import Q4_TILE_BYTES, make


def emit(M: int, N: int, K: int, output_dir: Path) -> tuple[Path, Path]:
    symbol = f"tl_q4_hmx_m{M}_n{N}_k{K}"
    prim_func = make(
        M,
        N,
        K,
        input_dtype="float32",
        output_dtype="float32",
    ).with_attr("global_symbol", symbol)
    target = determine_target("hexagon", return_object=True)
    with target:
        artifact = tilelang.lower(
            prim_func,
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    kernel_source = str(artifact.kernel_source)

    entry_match = re.search(r"(int32_t\s+\w+\([^)]*\))", kernel_source)
    high_water_match = re.search(
        r"tl_vtcm_shared_high_water\s*=\s*(\d+)u", kernel_source
    )
    if entry_match is None or high_water_match is None:
        raise RuntimeError("generated source is missing its entry or VTCM contract")

    kernel_entry = entry_match.group(1)
    high_water = int(high_water_match.group(1))
    embedded_symbol = f"{symbol}_embedded"
    embedded_entry = (
        f"int32_t {embedded_symbol}(void* vtcm_base, unsigned int vtcm_size, "
        "float* activation, uint8_t* weight, float* output)"
    )
    source = (
        "// Generated TileLang kernel plus its same-TU llama.cpp runtime bridge.\n"
        "#include <tl_templates/hexagon/tl_bridge.h>\n"
        "#define TL_Q4_0_PADDED_VTCM_INPUT 1\n"
        "#include <tl_templates/hexagon/qmatmul.h>\n\n"
        f"{kernel_source}\n"
        "#ifdef __cplusplus\nextern \"C\"\n#endif\n"
        f"{embedded_entry} {{\n"
        "  if (tl_bridge_enter(vtcm_base, vtcm_size) != 0) return -1;\n"
        f"  int32_t rc = {symbol}_kernel(activation, weight, output);\n"
        "  tl_bridge_exit();\n"
        "  return rc;\n"
        "}\n"
    )
    base = f"kernel_qmatmul_hmx_atoms_{M}x{N}x{K}"
    source_path = output_dir / f"{base}.cc"
    manifest_path = output_dir / f"{base}.manifest.json"

    manifest = {
        "schema_version": 2,
        "op": "matmul",
        "implementation": "explicit_hmx_atoms",
        "dequantization": "tilelang_scheduled_hvx_q4_tile_atoms",
        "weight_dtype": "q4_0",
        "activation_dtype": "float32",
        "output_dtype": "float32",
        "M": M,
        "N": N,
        "K": K,
        "entry": embedded_entry,
        "kernel_entry": kernel_entry,
        "vtcm_bytes": high_water,
        "weight_bytes": (N // 32) * (K // 32) * Q4_TILE_BYTES,
        "weight_format": {
            "shape": "[N/32][K/32][576] uint8",
            "quant": "tile[0:512] is [16 K-pairs][32 output channels]",
            "scale": "tile[512:576] is 32 little-endian FP16 scales",
            "source": "ggml-hexagon native repacked Q4_0 tile; pass weight directly",
        },
        "runtime_contract": {
            "hmx_session": "caller-owned",
            "vtcm": "caller-owned and bound with tl_bridge_enter",
            "dma": "caller-owned; registry double-buffers raw weight slices in VTCM",
            "weight_padding": "128B-aligned VTCM stage keeps 64B readable past final scale tail",
            "activation": f"contiguous [{M}][{K}] float32",
            "output": f"contiguous [{M}][{N}] float32",
            "call": (
                f"{embedded_symbol}(vtcm_base, vtcm_size, activation, weight, output)"
            ),
        },
        "lowering_contract": {
            "q4_atom": "tl_hexagon_q4_0_dequant_tile_32x32",
            "activation_atom": "tl_hexagon_hmx_pack_a_f32_pair_k32",
            "forbidden": [
                "T.gemm",
                "tl_hexagon_hmx_gemm",
                "tl_hexagon_hmx_pack_a",
                "tl_hexagon_hmx_pack_b",
                "tl.hexagon_q4_0_dequant",
                "tl_hexagon_hmx_dequant_q4_0",
                "qcm/scale repack cache",
            ],
            "hmx_protocol": [
                "acquire",
                "load_bias",
                "clear",
                "mma",
                "convert",
                "store",
                "release",
            ],
        },
        "note": "fixed HMX shape; emit one symbol per llama.cpp dispatch family",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return source_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()

    source_path, manifest_path = emit(args.m, args.n, args.k, args.output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"emitted {source_path.name} ({len(source_path.read_text().splitlines())} lines)")
    print(f"  entry : {manifest['entry']}")
    print(f"  vtcm  : {manifest['vtcm_bytes']} bytes")
    print(f"  weight: {manifest['weight_format']['shape']} (no repack cache)")


if __name__ == "__main__":
    main()
