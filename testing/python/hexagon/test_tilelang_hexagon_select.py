"""Lane-wise selects, comparisons and unary operators in Hexagon maps."""

import os

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T

DEVICE = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")

M, N = 64, 256


def _source(func):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


def _value(kind, a, i, j, dtype):
    zero = T.cast(0, dtype)
    if kind == "leaky_select":
        return T.Select(a[i, j] > zero, a[i, j], a[i, j] * T.cast(0.125, dtype))
    if kind == "leaky_if":
        return T.if_then_else(a[i, j] > zero, a[i, j], a[i, j] * T.cast(0.125, dtype))
    if kind == "causal":
        return T.if_then_else(j > i, -T.infinity(dtype), a[i, j])
    if kind == "band":
        return T.Select(T.And(a[i, j] > T.cast(-1, dtype), a[i, j] < T.cast(1, dtype)), a[i, j], zero)
    if kind == "wide_mask":
        # An fp32 comparison selects fp16 lanes.
        return T.Select(T.cast(a[i, j], "float32") * T.float32(3) > T.float32(1), a[i, j], zero)
    if kind == "abs":
        return T.abs(a[i, j])
    if kind == "not":
        return ~a[i, j]
    if kind == "shift_up":
        # The load past the last row is out of bounds: the branch stays guarded.
        return T.if_then_else(i + 1 < M, a[i + 1, j], zero)
    raise ValueError(kind)


def _map(kind, dtype):
    @T.prim_func
    def kernel(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((M, N), dtype)
            b = T.alloc_shared((M, N), dtype)
            T.copy(A, a)
            for i, j in T.Parallel(M, N):
                b[i, j] = _value(kind, a, i, j, dtype)
            T.copy(b, B)

    return kernel


def _reference(kind, a):
    import torch

    rows = torch.arange(M).unsqueeze(1)
    cols = torch.arange(N).unsqueeze(0)
    zero = torch.zeros((), dtype=a.dtype)
    if kind in ("leaky_select", "leaky_if"):
        return torch.where(a > 0, a, a * 0.125)
    if kind == "causal":
        return torch.where(cols > rows, torch.full((), float("-inf"), dtype=a.dtype), a)
    if kind == "band":
        return torch.where((a > -1) & (a < 1), a, zero)
    if kind == "wide_mask":
        return torch.where(a.float() * 3 > 1, a, zero)
    if kind == "abs":
        return torch.abs(a)
    if kind == "not":
        return ~a
    if kind == "shift_up":
        return torch.cat([a[1:], torch.zeros_like(a[:1])])
    raise ValueError(kind)


CASES = [
    ("leaky_select", "float16"),
    ("leaky_select", "float32"),
    ("leaky_if", "float16"),
    ("causal", "float16"),
    ("causal", "float32"),
    ("band", "float16"),
    ("wide_mask", "float16"),
    ("abs", "int8"),
    ("abs", "int32"),
    ("not", "int8"),
    ("shift_up", "float16"),
]


@pytest.mark.parametrize("kind, dtype", [c for c in CASES if c[0] != "shift_up"])
def test_masked_maps_select_lane_by_lane(kind, dtype):
    source = _source(_map(kind, dtype))
    lanes = 128 * 8 // tvm.DataType(dtype).bits
    vector = f"*({'half' if dtype == 'float16' else 'float' if dtype == 'float32' else dtype + '_t'}{lanes}*)(((" in source
    assert vector, source
    if kind != "not":
        assert "tl_vec_select(" in source
    assert "condval" not in source


def test_causal_masks_compare_a_lane_ramp():
    source = _source(_map("causal", "float16"))
    assert "tl_vec_ramp<int32_t, 64>(" in source


def test_loads_that_may_be_out_of_bounds_read_a_safe_address():
    source = _source(_map("shift_up", "float16"))
    # The last rows' loads would pass the end: they read the buffer's start
    # instead, and the select drops what they read. No branch.
    assert "half64 guarded = *(half64*)(((half*)a) + ((i_1 < 252) ? ((i_1 * 64) + 256) : 0));" in source
    assert "((i_1 < 252) ? guarded : " in source
    assert "condval" not in source


@DEVICE
@pytest.mark.parametrize("kind, dtype", CASES)
def test_masked_maps_match_torch(kind, dtype):
    import torch

    gen = torch.Generator().manual_seed(0)
    if dtype.startswith("int"):
        info = torch.iinfo(getattr(torch, dtype))
        a = torch.randint(info.min, info.max, (M, N), generator=gen, dtype=getattr(torch, dtype))
    else:
        a = (torch.randn(M, N, generator=gen) * 2).to(getattr(torch, dtype))
    kernel = tilelang.compile(_map(kind, dtype), out_idx=[1], target="hexagon")
    # HVX fp16 arithmetic flushes subnormal results (below 2^-14) to zero.
    atol = 2.0**-14 if dtype == "float16" else 0
    try:
        torch.testing.assert_close(kernel(a), _reference(kind, a), rtol=0, atol=atol)
    finally:
        kernel.adapter.close()
