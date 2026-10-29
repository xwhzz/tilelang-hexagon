"""The weight bundle: every tensor the device kernels read, in their layouts.

One file: an 8-byte magic, the header's length (little-endian u64), the
header (JSON), then the tensors, each starting on a 4 KB boundary. The device
loads each tensor into a resident buffer of its own (``load``), a kernel
argument. The host needs only the header: a copy of the file's first
``data_offset`` bytes serves.

Tensors stack layers along a leading dimension. Two writers fill it:

- ``quantize.py``, LFM2 in 4 bits: a projection [N, K] of a layer is S slices
  of its columns, SLICE each: signed levels [L, S, N, SLICE] packed in the
  panel layout (``tilelang.hexagon.intrinsics.panel``), so that ``block`` rows
  of a slice are contiguous, and fp16 scales per group of ``group`` columns,
  blocked by ``block`` rows to stream with them: scale ``[l, s, n // block,
  c // group, n % block]`` weighs column s * SLICE + c of row n, as
  ``T.gemm_blockscaled`` reads them.
- ``convert.py``, LFM2 or LFM3 in fp16: a projection is the HMX weight
  operand [L, K, N] in Crouton tiles (layout ``hmx_weight``).

    header["tensors"][name] = {"dtype", "shape", "offset", "nbytes", "layout"}
    header["quant"][name] = {"bits", "group", "block"}   (quantized ones)
"""

from __future__ import annotations

import json
import math
import struct

import torch

MAGIC = b"TLLFM2\x00\x01"
ALIGN = 4096

_TORCH = {"float32": torch.float32, "float16": torch.float16, "uint8": torch.uint8, "int8": torch.int8, "int32": torch.int32}
_BITS = {"float32": 32, "float16": 16, "uint8": 8, "int8": 8, "int32": 32, "int4": 4}


def _pad(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def write(path: str, config: dict, tensors: dict, quant: dict) -> dict:
    """Write `tensors` (name -> (data, dtype, shape, layout)): data is a tensor
    of the bytes (its own dtype or uint8), or a function that makes one when
    it is written, so that a large bundle need not be in memory at once."""
    header = {"config": config, "quant": quant, "tensors": {}}
    offset = 0
    for name, (_, dtype, shape, layout) in tensors.items():
        nbytes = math.prod(shape) * _BITS[dtype] // 8
        header["tensors"][name] = {"dtype": dtype, "shape": list(shape), "offset": offset, "nbytes": nbytes, "layout": layout}
        offset = _pad(offset + nbytes)
    head = json.dumps(header).encode()
    start = _pad(len(MAGIC) + 8 + len(head))
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<Q", len(head)) + head)
        for name, (data, *_) in tensors.items():
            t = header["tensors"][name]
            raw = (data() if callable(data) else data).contiguous().view(torch.uint8).reshape(-1)
            assert raw.numel() == t["nbytes"], (name, raw.numel(), t["nbytes"])
            f.seek(start + t["offset"])
            f.write(memoryview(raw.numpy()))
        f.truncate(start + offset)
    header["data_offset"] = start
    return header


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path} is not a weight bundle")
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


def load(session, path: str, device_path: str | None = None) -> dict:
    """The bundle's tensors in resident buffers of `session`, a buffer each
    (name -> DeviceBuffer), allocated the largest first: a buffer is under
    2 GiB, and the DSP's 4 GB of addresses take LFM3-1.7B's 3.5 GB as a
    buffer per tensor (``Session.buffer``).

    The tensors are read from the bundle at `device_path` on the device when
    given, so that the host needs only the header at `path`; else they are
    written from `path`."""
    header = read_header(path)
    tensors = header["tensors"]
    buffers = {}
    for name in sorted(tensors, key=lambda n: -tensors[n]["nbytes"]):
        t = tensors[name]
        buf = buffers[name] = session.buffer(t["nbytes"])
        if device_path is None:
            buf.write(read_tensor(path, header, name))
        elif buf.load(device_path, header["data_offset"] + t["offset"]) != t["nbytes"]:
            raise ValueError(f"{device_path} on the device ends inside {name}: not the bundle {path} heads")
    return {name: buffers[name] for name in tensors}
