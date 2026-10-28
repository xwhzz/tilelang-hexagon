"""Accuracy of the HVX math functions on the device, against torch in float64.

Vector min and max, which are exact, are checked here as well.
"""

import os

import pytest

import tilelang
from tilelang.hexagon import language as T

pytestmark = pytest.mark.skipif(os.environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")

LANES = 4096

# Function: torch reference, fp32 relative error bound, and input ranges for
# fp32 and fp16 (geometric if positive). The fp16 ranges keep inputs and
# results normal fp16 numbers.
CASES = {
    "exp": ("exp", 1e-5, (-20.0, 20.0), (-9.0, 10.0)),
    "exp2": ("exp2", 1e-5, (-30.0, 30.0), (-13.0, 15.0)),
    "log": ("log", 1e-6, (1e-6, 1e6), (1e-4, 6e4)),
    "log2": ("log2", 1e-6, (1e-6, 1e6), (1e-4, 6e4)),
    "sqrt": ("sqrt", 1e-6, (1e-6, 1e6), (1e-4, 6e4)),
    "rsqrt": ("rsqrt", 1e-6, (1e-6, 1e6), (1e-4, 6e4)),
    "tanh": ("tanh", 1e-5, (-8.0, 8.0), (-8.0, 8.0)),
    "sigmoid": ("sigmoid", 1e-5, (-16.0, 16.0), (-8.0, 8.0)),
    "erf": ("erf", 1e-5, (-4.0, 4.0), (-4.0, 4.0)),
    "abs": ("abs", 0.0, (-8.0, 8.0), (-8.0, 8.0)),
}


def _inputs(torch, lo, hi):
    if lo > 0:
        return torch.logspace(torch.log10(torch.tensor(lo)), torch.log10(torch.tensor(hi)), LANES, dtype=torch.float64)
    return torch.linspace(lo, hi, LANES, dtype=torch.float64)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("fn", sorted(CASES))
def test_math_matches_torch(fn, dtype):
    import torch

    ref, fp32_bound, fp32_range, fp16_range = CASES[fn]
    op = getattr(T, fn)

    @T.prim_func
    def math(A: T.Tensor((LANES,), dtype), B: T.Tensor((LANES,), dtype)):
        with T.Kernel(1):
            for j in T.Parallel(LANES):
                B[j] = op(A[j])

    kernel = tilelang.compile(math, out_idx=[1], target="hexagon")
    try:
        x = _inputs(torch, *(fp32_range if dtype == "float32" else fp16_range)).to(getattr(torch, dtype))
        expected = getattr(torch, ref)(x.double())
        absolute = (kernel(x).double() - expected).abs()
        # Near a zero of the function relative error is ill-conditioned, so
        # bound the absolute error there.
        near_zero = expected.abs() < 1e-3
        relative = absolute[~near_zero] / expected[~near_zero].abs()
        # fp16 results round once: half an fp16 ulp on top of the fp32 error.
        bound = fp32_bound if dtype == "float32" else 2.0**-11 + fp32_bound
        print(
            f"{fn:8s} {dtype}: max rel {relative.max().item():.3g}, max abs near 0 {absolute[near_zero].max().item() if near_zero.any() else 0:.3g}"
        )
        assert relative.max().item() <= bound
        assert not near_zero.any() or absolute[near_zero].max().item() <= 1e-5
    finally:
        kernel.adapter.close()


# The fast variants (tl.enable_fast_math): fp16-grade bounds, relative or, for
# the bounded functions, absolute, over the fp32 ranges above.
FAST = {
    "exp": ("rel", 1.2e-4),
    "exp2": ("rel", 1.2e-4),
    "log": ("rel", 3e-5),
    "log2": ("rel", 3e-5),
    "sqrt": ("rel", 2e-5),
    "rsqrt": ("rel", 2e-5),
    "tanh": ("abs", 2e-4),
    "sigmoid": ("rel", 2e-4),
    "erf": ("abs", 2e-4),
}


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("fn", sorted(FAST))
def test_fast_math_matches_torch(fn, dtype):
    import torch

    ref, _, fp32_range, fp16_range = CASES[fn]
    kind, bound = FAST[fn]
    op = getattr(T, fn)

    @T.prim_func
    def math(A: T.Tensor((LANES,), dtype), B: T.Tensor((LANES,), dtype)):
        with T.Kernel(1):
            for j in T.Parallel(LANES):
                B[j] = op(A[j])

    kernel = tilelang.compile(math, out_idx=[1], target="hexagon", pass_configs={"tl.enable_fast_math": True})
    try:
        x = _inputs(torch, *(fp32_range if dtype == "float32" else fp16_range)).to(getattr(torch, dtype))
        expected = getattr(torch, ref)(x.double())
        absolute = (kernel(x).double() - expected).abs()
        # fp16 results round once: half an fp16 ulp more.
        rounding = 0.0 if dtype == "float32" else 2.0**-11
        if kind == "abs":
            error = absolute.max().item()
            limit = bound + rounding
        else:
            near_zero = expected.abs() < 1e-3
            error = (absolute[~near_zero] / expected[~near_zero].abs()).max().item()
            limit = bound + rounding
            assert not near_zero.any() or absolute[near_zero].max().item() <= 1e-4
        print(f"{fn:8s} {dtype} fast: max {kind} {error:.3g}")
        assert error <= limit
    finally:
        kernel.adapter.close()


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_division_matches_torch(dtype, fast):
    import torch

    @T.prim_func
    def div(A: T.Tensor((LANES,), dtype), B: T.Tensor((LANES,), dtype), C: T.Tensor((LANES,), dtype)):
        with T.Kernel(1):
            for j in T.Parallel(LANES):
                C[j] = A[j] / B[j]

    kernel = tilelang.compile(div, out_idx=[2], target="hexagon", pass_configs={"tl.enable_fast_math": fast})
    try:
        g = torch.Generator().manual_seed(5)
        a = (torch.rand(LANES, generator=g, dtype=torch.float64) * 20 - 10).to(getattr(torch, dtype))
        b = (_inputs(torch, 1e-2, 1e2) * torch.where(torch.rand(LANES, generator=g) < 0.5, -1.0, 1.0)).to(getattr(torch, dtype))
        expected = a.double() / b.double()
        absolute = (kernel(a, b).double() - expected).abs()
        # Quotients below fp16's normal range round to its subnormal spacing.
        subnormal = expected.abs() < (2.0**-14 if dtype == "float16" else 0.0)
        relative = (absolute[~subnormal] / expected[~subnormal].abs()).max().item()
        bound = (2e-5 if fast else 1e-6) + (0.0 if dtype == "float32" else 2.0**-11)
        print(f"div {dtype} {'fast' if fast else 'accurate'}: max rel {relative:.3g}")
        assert relative <= bound
        assert not subnormal.any() or absolute[subnormal].max().item() <= 2.0**-25
    finally:
        kernel.adapter.close()


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_min_and_max_match_torch(dtype):
    import torch

    @T.prim_func
    def clamp(A: T.Tensor((LANES,), dtype), B: T.Tensor((LANES,), dtype), Lo: T.Tensor((LANES,), dtype), Hi: T.Tensor((LANES,), dtype)):
        with T.Kernel(1):
            for j in T.Parallel(LANES):
                Lo[j] = T.min(A[j], B[j])
                Hi[j] = T.max(A[j], B[j])

    kernel = tilelang.compile(clamp, out_idx=[2, 3], target="hexagon")
    try:
        torch.manual_seed(0)
        if dtype == "int8":
            a, b = (torch.randint(-128, 128, (LANES,), dtype=torch.int8) for _ in range(2))
        else:
            a, b = (torch.randn(LANES).to(getattr(torch, dtype)) for _ in range(2))
        lo, hi = kernel(a, b)
        torch.testing.assert_close(lo, torch.minimum(a, b), rtol=0, atol=0)
        torch.testing.assert_close(hi, torch.maximum(a, b), rtol=0, atol=0)
    finally:
        kernel.adapter.close()


@pytest.mark.parametrize("src", ["lane", "uint8", "int16", "int32"])
def test_integer_lanes_convert_to_fp32(src):
    import torch

    # LLVM converts lanes it proves non-negative (a lane index, a zero
    # extension) as unsigned, which hexagon-clang 19 lowers to IEEE fp adds:
    # the lanes read 0, or the backend fails on the select.
    dtype = "int32" if src == "lane" else src

    @T.prim_func
    def convert(A: T.Tensor((LANES,), dtype), Out: T.Tensor((2, LANES), "float32")):
        with T.Kernel(1):
            f = T.alloc_fragment((LANES,), "float32")
            for j in T.Parallel(LANES):
                f[j] = T.Cast("float32", j) if src == "lane" else T.Cast("float32", A[j])
            for j in T.Parallel(LANES):
                Out[0, j] = f[j]
                Out[1, j] = T.if_then_else(A[j] == A[0], T.Cast("float32", j), T.float32(-1))

    kernel = tilelang.compile(convert, out_idx=[1], target="hexagon")
    try:
        torch.manual_seed(0)
        top = {"int32": 1 << 30, "uint8": 256, "int16": 1 << 15}[dtype]
        a = torch.randint(0, top, (LANES,), dtype=torch.int64)
        a[::7] = a[0]
        a = a.to(getattr(torch, dtype))
        out = kernel(a)
        lanes = torch.arange(LANES, dtype=torch.float32)
        torch.testing.assert_close(out[0], lanes if src == "lane" else a.to(torch.float64).float(), rtol=0, atol=0)
        torch.testing.assert_close(out[1], torch.where(a == a[0], lanes, torch.tensor(-1.0)), rtol=0, atol=0)
    finally:
        kernel.adapter.close()
