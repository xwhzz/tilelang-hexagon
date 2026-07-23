"""Emit staged Q4_0/HMX kernels for a host-owned worker/DMA schedule.

TileLang owns the pack, dequant, Crouton layouts, and explicit HMX protocol.
The embedding runtime may run the dequant stage over disjoint tile ranges on
its existing workers, then call the single-thread HMX stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tilelang
from tilelang.utils.target import determine_target

from examples.hexagon.example_qmatmul_kstream import (
    make_staged_dequant,
    make_staged_hmx,
    make_staged_pack,
)


M = 32
N = 256
PADDED_Q4_TILE_BYTES = 640


def _lower(func, symbol: str) -> str:
    target = determine_target("hexagon", return_object=True)
    with target:
        artifact = tilelang.lower(
            func.with_attr("global_symbol", symbol),
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    return str(artifact.kernel_source)


def emit(K: int, output_dir: Path) -> tuple[Path, Path]:
    if K <= 0 or K % 32:
        raise ValueError("K must be a positive multiple of 32")

    base_symbol = f"tl_q4_hmx_staged_m{M}_n{N}_k{K}"
    symbols = {
        "pack": f"{base_symbol}_pack",
        "dequant": f"{base_symbol}_dequant",
        "compute": f"{base_symbol}_compute",
    }
    kernels = [
        _lower(make_staged_pack(M, K), symbols["pack"]),
        _lower(
            make_staged_dequant(N, K, PADDED_Q4_TILE_BYTES),
            symbols["dequant"],
        ),
        _lower(make_staged_hmx(M, N, K), symbols["compute"]),
    ]
    source = (
        "// Generated TileLang stages for llama.cpp-owned DMA and workers.\n"
        "#include <tl_templates/hexagon/tl_bridge.h>\n"
        "#define TL_Q4_0_PADDED_VTCM_INPUT 1\n"
        "#include <tl_templates/hexagon/qmatmul.h>\n\n"
        + "\n".join(kernels)
    )

    base = f"kernel_qmatmul_hmx_staged_{M}x{N}x{K}"
    source_path = output_dir / f"{base}.cc"
    manifest_path = output_dir / f"{base}.manifest.json"
    entries = {name: f"{symbol}_kernel" for name, symbol in symbols.items()}
    kt = K // 32
    manifest = {
        "schema_version": 2,
        "op": "matmul",
        "implementation": "staged_explicit_hmx_atoms",
        "M": M,
        "N": N,
        "K": K,
        "entries": entries,
        "dequantization": "parallel_range_of_hvx_q4_tile_atoms",
        "padded_weight_format": {
            "shape": f"[{N // 32}][{kt}][{PADDED_Q4_TILE_BYTES}] uint8",
            "dma": "2D 576B -> 640B rows",
        },
        "vtcm": {
            "activation_bytes": M * K * 2,
            "dequant_bytes": N * K * 2,
            "output_crouton_bytes": M * 32 * 2,
            "bias_bytes": 64 * 4,
            "raw_weight_stage_bytes": (N // 32) * kt * PADDED_Q4_TILE_BYTES,
            "raw_weight_stages": 2,
        },
        "runtime_contract": {
            "workers": "caller-owned synchronous parallel_for",
            "dma": "caller-owned ordered async 2D queue",
            "hmx": "caller locks compute resource; compute stage owns accumulator protocol",
        },
        "lowering_contract": {
            "q4_atom": "tl_hexagon_q4_0_dequant_tile_32x32",
            "activation_atom": "tl_hexagon_hmx_pack_a_f32_pair_k32",
            "hmx_atom": "tl_hexagon_hmx_mma_atom",
            "layout": "TileLang T.Layout for activation/weight/output Croutons",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return source_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    source, manifest = emit(args.k, args.output_dir)
    print(f"emitted {source.name}")
    print(f"  manifest: {manifest.name}")


if __name__ == "__main__":
    main()
