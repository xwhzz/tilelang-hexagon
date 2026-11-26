"""Host-side packing of dense Hexagon kernel buffers in their declared storage order."""

from math import prod

import numpy as np
import torch


def storage_layout(param) -> tuple[int, tuple[int, ...] | None]:
    """Physical span in elements and axis order for a contiguous permutation."""
    shape, strides = param.shape, param.strides
    if not all(isinstance(n, int) for n in shape):
        raise NotImplementedError("Hexagon adapter: dynamic shapes not supported yet")
    if not strides:
        return prod(shape), ()
    if len(strides) != len(shape) or not all(isinstance(s, int) and s >= 0 for s in strides):
        raise NotImplementedError("Hexagon buffer transfers require static, nonnegative element strides for every axis")
    if 0 in shape:
        return 0, ()
    axes = sorted(range(len(shape)), key=strides.__getitem__, reverse=True)
    extent = 1
    for axis in reversed(axes):
        if shape[axis] > 1 and strides[axis] != extent:
            # Resident arguments and raw bytes can name gapped and overlapping layouts.
            return 1 + sum((n - 1) * s for n, s in zip(shape, strides)), None
        extent *= shape[axis]
    active = [axis for axis in axes if shape[axis] > 1]
    return extent, tuple(axes) if active != sorted(active) else ()


def buffer_bytes(data) -> bytes:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, torch.Tensor):
        data = data.detach().to("cpu").contiguous()
        if data.dtype == torch.bfloat16:
            data = data.view(torch.int16)
        return data.numpy().tobytes()
    return np.ascontiguousarray(data).tobytes()


def pack_buffer(data, param, plan, kernel_name) -> bytes:
    """Pack logical elements in storage order. Raw and sub-byte buffers are already packed."""
    if plan.storage_axes != () and not isinstance(data, (bytes, bytearray, memoryview)) and str(param.dtype) not in ("int4", "uint4"):
        if plan.storage_axes is None:
            raise NotImplementedError("pass raw bytes or a resident buffer for non-dense Hexagon inputs")
        if not isinstance(data, torch.Tensor):
            data = np.asarray(data)
        if tuple(data.shape) != tuple(param.shape):
            raise ValueError(f"expected shape {tuple(param.shape)} for strided input, got {tuple(data.shape)}")
        data = data.permute(plan.storage_axes) if isinstance(data, torch.Tensor) else data.transpose(plan.storage_axes)
    raw = buffer_bytes(data)
    if len(raw) != plan.nbytes:
        raise ValueError(f"argument {plan.index} of {kernel_name} has {len(raw)} bytes, the kernel takes {plan.nbytes}")
    return raw


def unpack_buffer(raw, param):
    """Expose the returned storage as a logical tensor, except for packed 4-bit bytes."""
    dtype = str(param.dtype)
    packed = dtype in ("int4", "uint4")
    torch_dtype = torch.uint8 if packed else getattr(torch, dtype)
    flat = torch.frombuffer(bytearray(raw), dtype=torch_dtype) if raw else torch.empty(0, dtype=torch_dtype)
    if packed:
        return flat
    if param.strides:
        return flat.as_strided(param.shape, param.strides)
    return flat.reshape(param.shape)
