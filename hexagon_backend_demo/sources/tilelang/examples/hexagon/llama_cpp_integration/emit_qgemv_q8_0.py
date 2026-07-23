"""Emit fixed-K Q8_0 dot atoms for the llama.cpp Hexagon embedding example.

The emitted entry is a normal TileLang Hexagon kernel body.  It calls the
``Q8GemvIntrinEmitter`` instruction atom and has no FastRPC/session wrapper, so
the model integration can co-compile it directly into ``libggml-htp-v79.so``.

    python emit_qgemv_q8_0.py                 # K=2048 and K=8192
    python emit_qgemv_q8_0.py --k 2048
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import tilelang
from tilelang.hexagon.qgemv import Q8_0_TILED_LAYOUT, make_q8_0_dot_prim_func


def emit(k: int, output_dir: Path, *, with_bias: bool) -> None:
    layout = Q8_0_TILED_LAYOUT
    layout.validate_k(k)
    symbol = f"qgemv_q8_0_k{k}" + ("" if with_bias else "_nobias")
    kernel = tilelang.compile(
        make_q8_0_dot_prim_func(k, symbol=symbol, with_bias=with_bias),
        out_idx=[3 if with_bias else 2],
        target="hexagon",
    )
    source = kernel.get_kernel_source()
    entry_match = re.search(r"(int32_t\s+\w+\([^)]*\))", source)
    if entry_match is None:
        raise RuntimeError("could not find generated Hexagon entry signature")

    base = output_dir / f"kernel_{symbol}"
    manifest = {
        "op": "q8_0_dot_32x1",
        "K": k,
        "N": layout.block_n,
        "bias": with_bias,
        "entry": entry_match.group(1),
        "source_weight_tile_bytes": layout.source_tile_bytes,
        "staged_tile_bytes": layout.staged_tile_bytes,
        "staged_weight_bytes": layout.staged_weight_bytes(k),
        "staged_activation_bytes": layout.staged_activation_bytes(k),
        "layout": "ggml-hexagon q8_0_tiled after 2D-DMA padding",
        "ownership": {
            "tilelang": "HVX signed-int8 dot + scale accumulation",
            "embedding_runtime": "DMA, VTCM buffers, activation quantization, worker pool",
        },
    }
    base.with_suffix(".c").write_text(source, encoding="ascii")
    base.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii"
    )
    print(f"emitted {base.name}.c")
    print(f"  entry : {manifest['entry']}")
    print(f"  staged: {manifest['staged_weight_bytes']} B weight + activation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for k in args.k:
        emit(k, args.output_dir, with_bias=True)
        emit(k, args.output_dir, with_bias=False)


if __name__ == "__main__":
    main()
