"""Integer HMX output conversion from an ordinary GEMM and narrowing map."""

import pytest
import torch

import tilelang
from tilelang.hexagon import language as T
from tilelang.hexagon.intrinsics.panel import make_hvx_panel_layout, pack_panel
from test_tilelang_hexagon_gemm_int import DEVICE, _lower


def _kernel(n=64, *, paired=True, workers=1, scope="local.fragment", mode="fused", offset=0):
    m, k = 128, 1056
    batch = 2 * workers
    rows = m - (mode == "partial")
    cols = n + offset

    @T.prim_func
    def kernel(
        X: T.Tensor((batch, m, k), "uint8"),
        W: T.Tensor((batch, n, k), "int8"),
        P: T.Tensor((batch, 2, n), "int32"),
        Y: T.Tensor((batch, m, cols), "int16"),
        Raw: T.Tensor((batch, m, n), "int32"),
    ):
        with T.Kernel(batch, num_workers=workers) as w:
            a = T.alloc_shared((m, k), "uint8")
            b = T.alloc_shared((n, k), "int8")
            ci = T.sblock_alloc_buffer((m, n), "int32", scope=scope)
            y = T.alloc_shared((m, cols), "int16")
            bias = T.alloc_shared((n,), "int32")
            shift = T.alloc_shared((n,), "int32")
            T.annotate_layout({W: make_hvx_panel_layout(W, paired=paired), b: make_hvx_panel_layout(b, paired=paired)})
            if mode == "hvx":
                T.annotate_layout({a: T.Layout(a.shape, lambda i, j: [i, j])})
            if mode == "strided":
                T.annotate_layout({y: T.Layout(y.shape, lambda i, j: [j, i])})
            T.copy(X[w, :, :], a)
            T.copy(W[w, :, :], b)
            T.copy(P[w, 0, :], bias)
            T.copy(P[w, 1, :], shift)
            if offset or mode == "partial":
                T.clear(y)
            if mode == "accumulate":
                T.clear(ci)
            T.gemm(a, b, ci, transpose_B=True, clear_accum=mode != "accumulate")
            for i, j in T.Parallel(rows, n):
                if mode == "global_shift":
                    y[i, j + offset] = T.Cast("int16", (ci[i, j] + bias[j]) >> P[w, 1, j])
                elif mode == "permuted":
                    y[i, j + offset] = T.Cast("int16", (ci[i, (j + 1) % n] + bias[j]) >> shift[j])
                elif mode == "row_bias":
                    y[i, j + offset] = T.Cast("int16", (ci[i, j] + bias[i % n]) >> shift[j])
                elif mode == "row_shift":
                    y[i, j + offset] = T.Cast("int16", (ci[i, j] + bias[j]) >> shift[i % n])
                elif mode == "subtract":
                    y[i, j + offset] = T.Cast("int16", (ci[i, j] - bias[j]) >> shift[j])
                elif mode == "guarded":
                    if i != j:
                        y[i, j + offset] = T.Cast("int16", (ci[i, j] + bias[j]) >> shift[j])
                else:
                    y[i, j + offset] = T.Cast("int16", (ci[i, j] + bias[j]) >> shift[j])
            if mode == "escape":
                Raw[w, 0, 0] = T.call_extern("int32", "observe_acc", ci.access_ptr("r"))
            if mode == "extra":
                T.copy(ci, Raw[w, :, :])
            if mode == "mixed":
                other = T.alloc_shared((m, n), "int32")
                T.gemm(a, b, other, transpose_B=True, clear_accum=True)
                T.copy(other, Raw[w, :, :])
            T.copy(y, Y[w, :, :])

    return kernel


@pytest.mark.parametrize("n,paired", [(32, False), (64, False), (128, True)])
@pytest.mark.parametrize("scope", ["shared.dyn", "local.fragment", "local"])
def test_requant_lowering(n, paired, scope):
    src = _lower(_kernel(n, paired=paired, scope=scope, offset=3)).kernel_source
    assert f"tl_hexagon_hmx_gemm_int_requant<false, {str(paired).lower()}>" in src
    assert "tl_hexagon_hmx_gemm_int<" not in src and "void* ci =" not in src and " ci[" not in src


@pytest.mark.parametrize(
    "mode",
    ["extra", "accumulate", "row_bias", "row_shift", "subtract", "partial", "guarded", "global_shift", "strided", "permuted", "escape"],
)
def test_requant_near_miss(mode):
    src = _lower(_kernel(mode=mode, scope="shared.dyn" if mode == "permuted" else "local.fragment")).kernel_source
    assert "tl_hexagon_hmx_gemm_int<false, true>" in src
    assert "tl_hexagon_hmx_gemm_int_requant<" not in src


def test_requant_does_not_steal_row_major_hvx_gemm():
    src = _lower(_kernel(mode="hvx", paired=False)).kernel_source
    assert "tl_hvx_gemm_int<" in src and "tl_hexagon_hmx_gemm_int_requant<" not in src


def test_other_parallel_local_access_still_requires_a_fragment():
    with pytest.raises(ValueError, match="Local buffer .*ci.* is indexed by T.Parallel"):
        _lower(_kernel(scope="local", mode="extra"))


def test_requant_workers_invalidate_exact_banks():
    src = _lower(_kernel(mode="mixed", workers=2)).kernel_source
    assert "tl_hexagon_hmx_gemm_int_requant<true, true>" in src
    assert next(line for line in src.splitlines() if "tl_hexagon_hmx_gemm_int<true, true>" in line).endswith(", true);")


@DEVICE
@pytest.mark.parametrize("n,paired,workers", [(96, False, 1), (128, True, 1), (96, False, 2), (128, True, 2)])
@pytest.mark.parametrize("wide_shift", [False, True])
def test_requant_matches_exact_and_cpu(n, paired, workers, wide_shift):
    batch, m, k = workers * 2, 128, 1056
    rng = torch.Generator().manual_seed(432 + n)
    a = torch.randint(0, 256, (batch, m, k), generator=rng, dtype=torch.uint8)
    w = torch.randint(-128, 128, (batch, n, k), generator=rng, dtype=torch.int8)
    a[:, 0] = 255
    w[:, 0::8], w[:, 1::8] = 127, -128
    dot = a.long() @ w.long().transpose(1, 2)
    params = torch.empty((batch, 2, n), dtype=torch.int32)
    bias, shift = params.unbind(1)
    bias.random_(-(2**31), 2**31, generator=rng)
    bias[:, 0::8], bias[:, 1::8] = 2**31 - 1, -(2**31)
    shift.random_(0, 17, generator=rng)
    shift[:, 0::8], shift[:, 1::8], shift[:, 2::8] = 0, 16, 8
    if wide_shift:
        # Different column groups and worker jobs take different bank paths.
        shift[::2, 32:64:2], shift[::2, 33:64:2] = 17, 31
    # Put the second activation row exactly on either side of both int16 limits.
    for j, value in enumerate((-32769, -32768, 32767, 32768), 3):
        bias[:, j] = ((value << shift[:, j].long()) - dot[:, 1, j]).int()
    reference = ((dot + bias[:, None, :]).int() >> shift[:, None, :]).short()
    packed = pack_panel(w, 8, paired=paired)
    results = []
    for mode in ("mixed", "extra"):
        func = _kernel(n, paired=paired, workers=workers, mode=mode, offset=3)
        compiled = tilelang.compile(func, out_idx=[3, 4], target="hexagon")
        try:
            y, raw = compiled(a, packed, params)
        finally:
            compiled.adapter.close()
        assert torch.equal(raw, dot.int())
        assert torch.count_nonzero(y[:, :, :3]) == 0
        assert torch.equal(y[:, :, 3:], reference)
        results.append(y)
    assert torch.equal(*results)


def test_requant_subregions_keep_independent_column_origins():
    @T.prim_func
    def region(
        A: T.Tensor((64, 32), "uint8"), W: T.Tensor((32, 32), "int8"), P: T.Tensor((2, 64), "int32"), Y: T.Tensor((2, 68, 40), "int16")
    ):
        with T.Kernel(2) as batch:
            a = T.alloc_shared((64, 32), "uint8")
            b = T.alloc_shared((32, 32), "int8")
            ci = T.alloc_shared((128, 96), "int32")
            bias = T.alloc_shared((64,), "int32")
            shift = T.alloc_shared((64,), "int32")
            T.annotate_layout({W: make_hvx_panel_layout(W)})
            T.copy(A, a)
            T.copy(W, b)
            T.copy(P[0, :], bias)
            T.copy(P[1, :], shift)
            T.gemm(a, b, ci[64:, 32:64], transpose_B=True, clear_accum=True)
            for i, j in T.Parallel(64, 32):
                Y[batch, i + 2, j + 5] = T.Cast("int16", (bias[j + 7] + ci[i + 64, j + 32]) >> shift[j + 11])

    src = _lower(region).kernel_source
    call = next(line for line in src.splitlines() if "tl_hexagon_hmx_gemm_int_requant<" in line)
    assert "Y[" in call and "40," in call
    assert "bias)[7]" in call and "shift)[11]" in call
    assert "void* ci =" not in src
