"""The LFM2 weight bundle: every tensor the device kernels read, in their layouts.

One file: an 8-byte magic, the header's length (little-endian u64), the
header (JSON), then the tensors, each starting on a 4 KB boundary. The device
loads the file into one resident buffer, and each kernel argument is a slice
of it (``DeviceBuffer.at``).

Tensors stack layers along a leading dimension. A quantized projection
[N, K] of a layer is S slices of its columns, SLICE each (``quantize.py``):
signed levels [L, S, N, SLICE] packed in the panel layout
(``tilelang.hexagon.intrinsics.panel``), so that ``block`` rows of a slice are
contiguous, and fp16 scales per group of ``group`` columns, blocked by
``block`` rows to stream with them: scale ``[l, s, n // block, c // group,
n % block]`` weighs column s * SLICE + c of row n, as ``T.gemm_blockscaled``
reads them.

    header["tensors"][name] = {"dtype", "shape", "offset", "nbytes", "layout"}
    header["quant"][name] = {"bits", "group", "block"}   (quantized ones)
"""

from __future__ import annotations

import json
import struct

import torch

MAGIC = b"TLLFM2\x00\x01"
ALIGN = 4096

_TORCH = {"float32": torch.float32, "float16": torch.float16, "uint8": torch.uint8, "int8": torch.int8, "int32": torch.int32}


def _pad(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def write(path: str, config: dict, tensors: dict, quant: dict) -> dict:
    """Write `tensors` (name -> (bytes tensor or tensor, dtype, shape, layout))."""
    header = {"config": config, "quant": quant, "tensors": {}}
    blobs = []
    offset = 0
    for name, (data, dtype, shape, layout) in tensors.items():
        raw = data.contiguous().view(torch.uint8).reshape(-1) if isinstance(data, torch.Tensor) else data
        header["tensors"][name] = {"dtype": dtype, "shape": list(shape), "offset": offset, "nbytes": int(raw.numel()), "layout": layout}
        blobs.append((offset, raw))
        offset = _pad(offset + raw.numel())
    head = json.dumps(header).encode()
    start = _pad(len(MAGIC) + 8 + len(head))
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<Q", len(head)) + head)
        f.write(bytes(start - f.tell()))
        for off, raw in blobs:
            f.seek(start + off)
            f.write(raw.numpy().tobytes())
        f.truncate(start + offset)
    header["data_offset"] = start
    return header


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path} is not an LFM2 bundle")
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header["data_offset"] = _pad(len(MAGIC) + 8 + n)
    return header


def read_tensor(path: str, header: dict, name: str) -> torch.Tensor:
    """A tensor's bytes from the bundle: uint8, or its dtype when it has one."""
    t = header["tensors"][name]
    with open(path, "rb") as f:
        f.seek(header["data_offset"] + t["offset"])
        raw = torch.frombuffer(bytearray(f.read(t["nbytes"])), dtype=torch.uint8)
    if t["dtype"] in _TORCH and t["layout"] == "row_major":
        return raw.view(_TORCH[t["dtype"]]).reshape(t["shape"])
    return raw
