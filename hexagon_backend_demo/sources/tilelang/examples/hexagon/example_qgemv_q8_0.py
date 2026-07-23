"""Q8_0 decode dot on Hexagon, expressed as a TileLang instruction atom.

The kernel consumes the same 1152-byte-per-K-block staged layout used by
ggml-hexagon after its 2D DMA.  This example validates the atom independently;
``llama_cpp_integration/emit_qgemv_q8_0.py`` embeds the identical PrimFunc in a
real model while leaving ggml's DMA, activation quantization, and six-worker
scheduler intact.

    python examples/hexagon/example_qgemv_q8_0.py --k 2048
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import tilelang
from tilelang.hexagon.qgemv import Q8_0_TILED_LAYOUT, make_q8_0_dot_prim_func


def pack_weight_tile(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Pack ``q[32,K]`` + ``scales[32,K/32]`` into staged Q8_0 tiles."""

    layout = Q8_0_TILED_LAYOUT
    if q.dtype != np.int8 or q.ndim != 2 or q.shape[0] != layout.block_n:
        raise ValueError("q must be int8[32,K]")
    k = q.shape[1]
    layout.validate_k(k)
    if scales.shape != (layout.block_n, k // layout.block_k):
        raise ValueError(f"scales must have shape (32, {k // layout.block_k})")
    scales = np.asarray(scales, dtype=np.float16)

    staged = np.zeros(layout.staged_weight_bytes(k), dtype=np.uint8)
    for kt in range(layout.k_tiles(k)):
        tile = staged[
            kt * layout.staged_tile_bytes : (kt + 1) * layout.staged_tile_bytes
        ]
        block = q[:, kt * layout.block_k : (kt + 1) * layout.block_k]
        for pair in range(layout.block_k // 2):
            tile[pair * 64 : (pair + 1) * 64] = block[
                :, 2 * pair : 2 * pair + 2
            ].reshape(-1).view(np.uint8)
        tile[layout.quant_bytes : layout.source_tile_bytes] = np.ascontiguousarray(
            scales[:, kt]
        ).view(np.uint8)
    return staged


def pack_activation_tile(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Replicate ``q[K]`` into the eight HVX vectors consumed by vrmpyacc."""

    layout = Q8_0_TILED_LAYOUT
    q = np.asarray(q, dtype=np.int8)
    if q.ndim != 1:
        raise ValueError("q must be int8[K]")
    k = q.shape[0]
    layout.validate_k(k)
    scales = np.asarray(scales, dtype=np.float16)
    if scales.shape != (layout.k_tiles(k),):
        raise ValueError(f"scales must have shape ({layout.k_tiles(k)},)")

    staged = np.zeros(layout.staged_activation_bytes(k), dtype=np.uint8)
    for kt in range(layout.k_tiles(k)):
        tile = staged[
            kt * layout.staged_tile_bytes : (kt + 1) * layout.staged_tile_bytes
        ]
        block = q[kt * layout.block_k : (kt + 1) * layout.block_k]
        for group in range(8):
            replicated = np.tile(block[group * 4 : group * 4 + 4], layout.block_n)
            tile[group * layout.hvx_bytes : (group + 1) * layout.hvx_bytes] = (
                replicated.view(np.uint8)
            )
        tile[layout.quant_bytes : layout.staged_tile_bytes] = np.full(
            64, scales[kt], dtype=np.float16
        ).view(np.uint8)
    return staged


def reference(
    weight_q: np.ndarray,
    weight_scales: np.ndarray,
    activation_q: np.ndarray,
    activation_scales: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    layout = Q8_0_TILED_LAYOUT
    out = np.asarray(bias, dtype=np.float32).copy()
    for kt in range(layout.k_tiles(weight_q.shape[1])):
        k0 = kt * layout.block_k
        dot = weight_q[:, k0 : k0 + layout.block_k].astype(np.int32) @ activation_q[
            k0 : k0 + layout.block_k
        ].astype(np.int32)
        out += dot.astype(np.float32) * weight_scales[:, kt].astype(
            np.float32
        ) * np.float32(activation_scales[kt])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--no-bias", action="store_true")
    args = parser.parse_args()
    k = args.k
    layout = Q8_0_TILED_LAYOUT
    layout.validate_k(k)

    rng = np.random.default_rng(0)
    weight_q = rng.integers(-96, 96, size=(32, k), dtype=np.int8)
    activation_q = rng.integers(-96, 96, size=(k,), dtype=np.int8)
    weight_scales = (rng.random((32, k // 32)) * 0.02 + 0.001).astype(np.float16)
    activation_scales = (rng.random(k // 32) * 0.02 + 0.001).astype(np.float16)
    bias = (
        np.zeros(32, dtype=np.float32)
        if args.no_bias
        else rng.standard_normal(32).astype(np.float32)
    )

    packed_weight = pack_weight_tile(weight_q, weight_scales)
    packed_activation = pack_activation_tile(activation_q, activation_scales)
    ref = reference(weight_q, weight_scales, activation_q, activation_scales, bias)

    symbol = f"qgemv_q8_0_k{k}" + ("_nobias" if args.no_bias else "")
    prim = make_q8_0_dot_prim_func(k, symbol=symbol, with_bias=not args.no_bias)
    kernel = tilelang.compile(prim, out_idx=[2 if args.no_bias else 3], target="hexagon")
    inputs = [torch.from_numpy(packed_weight), torch.from_numpy(packed_activation)]
    if not args.no_bias:
        inputs.append(torch.from_numpy(bias))
    out = kernel(*inputs).float()
    ref_t = torch.from_numpy(ref)
    max_abs = (out - ref_t).abs().max().item()
    max_rel = ((out - ref_t).abs() / ref_t.abs().clamp_min(1.0)).max().item()
    passed = torch.allclose(out, ref_t, rtol=3e-4, atol=3e-3)
    print(
        f"Q8_0 dot 32x{k}: max_abs={max_abs:.6g} max_rel={max_rel:.6g} "
        f"({'PASS' if passed else 'FAIL'})"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
