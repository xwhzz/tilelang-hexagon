"""Hexagon maps vectorize through the standard lowering onto whole HVX registers."""

import re

import pytest

import tilelang
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon import language as T


def _source(func, fast_math=False):
    target = determine_target("hexagon", return_object=True)
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3, config={"tl.enable_fast_math": fast_math}):
        return str(tilelang.lower(func, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source)


@pytest.mark.parametrize("dtype,vector", [("float16", "half64"), ("float32", "float32"), ("int8", "int8_t128")])
def test_maps_use_whole_hvx_registers(dtype, vector):
    @T.prim_func
    def add(A: T.Tensor((4, 512), dtype), B: T.Tensor((4, 512), dtype), C: T.Tensor((4, 512), dtype)):
        with T.Kernel(4) as bx:
            for j in T.Parallel(512):
                C[bx, j] = A[bx, j] + B[bx, j]

    source = _source(add)
    # One 1024-bit register per vector, as clang ext_vector arithmetic.
    assert f"*({vector}*)(C + " in source
    assert f"(*({vector}*)(A + " in source


def test_a_narrow_input_fills_whole_registers():
    @T.prim_func
    def widen(A: T.Tensor((4, 512), "uint8"), B: T.Tensor((4, 512), "float16")):
        with T.Kernel(4) as bx:
            for j in T.Parallel(512):
                B[bx, j] = T.Cast("float16", A[bx, j])

    source = _source(widen)
    # 128 uint8 lanes fill one register; the fp16 result takes two.
    assert "*(uint8_t128*)(A + " in source
    assert "((half128)*(uint8_t128*)" in source
    assert "uint8_t64" not in source


def test_a_loop_too_short_for_a_register_is_scalar():
    @T.prim_func
    def widen(A: T.Tensor((64,), "uint8"), B: T.Tensor((64,), "float16")):
        with T.Kernel(1):
            for j in T.Parallel(64):
                B[j] = T.Cast("float16", A[j])

    source = _source(widen)
    # 64 uint8 lanes are half a register, which HVX cannot operate on.
    assert "uint8_t64" not in source
    assert "half64" not in source


@pytest.mark.parametrize("dtype,vector", [("float32", "float32"), ("float16", "half64")])
def test_vector_division_uses_the_hvx_reciprocal(dtype, vector):
    @T.prim_func
    def divide(A: T.Tensor((4, 256), dtype), B: T.Tensor((4, 256), dtype), C: T.Tensor((4, 256), dtype)):
        with T.Kernel(4) as bx:
            for j in T.Parallel(256):
                C[bx, j] = A[bx, j] / B[bx, j]
            # A scalar division stays a C division.
            C[bx, 0] = A[bx, 0] / B[bx, 1]

    source = _source(divide)
    assert f"tl_hvx_div(*({vector}*)(A + " in source
    assert "(A[(bx * 256)] / B[((bx * 256) + 1)])" in source


@pytest.mark.parametrize("dtype,vector", [("float32", "float32"), ("float16", "half64"), ("int8", "int8_t128")])
def test_vector_min_and_max_are_hvx_instructions(dtype, vector):
    @T.prim_func
    def clamp(A: T.Tensor((4, 256), dtype), B: T.Tensor((4, 256), dtype)):
        with T.Kernel(4) as bx:
            for j in T.Parallel(256):
                B[bx, j] = T.min(T.max(A[bx, j], T.cast(0, dtype)), T.cast(1, dtype))
            # A scalar min stays a C conditional.
            B[bx, 0] = T.min(A[bx, 1], A[bx, 2])

    source = _source(clamp)
    assert f"tl_vec_min(tl_vec_max(*({vector}*)(A + " in source
    assert ") < (" in source


# DSL function: the hvx_math.h function it prints as.
MATH = {fn: f"tl_hvx_{fn}" for fn in ["exp", "exp2", "log", "log2", "sqrt", "rsqrt", "tanh", "sigmoid", "erf"]}
MATH["abs"] = "tl_hvx_fabs"


@pytest.mark.parametrize("fn", sorted(MATH))
@pytest.mark.parametrize("dtype,vector", [("float32", "float32"), ("float16", "half64")])
def test_math_vectorizes_to_hvx_functions(fn, dtype, vector):
    op = getattr(T, fn)

    @T.prim_func
    def math(A: T.Tensor((4, 256), dtype), B: T.Tensor((4, 256), dtype)):
        with T.Kernel(4) as bx:
            for j in T.Parallel(256):
                B[bx, j] = op(A[bx, j])
            # A scalar runs the same HVX kernel.
            B[bx, 0] = op(A[bx, 1])

    source = _source(math)
    assert f"{MATH[fn]}(*({vector}*)(A + " in source
    assert f"{MATH[fn]}(A[" in source
    # No libm call, scalar or vector.
    for libm in ("expf", "exp2f", "logf", "log2f", "sqrtf", "tanhf", "erff", "fabsf"):
        assert f"{libm}(" not in source


@pytest.mark.parametrize("dtype,vector", [("float32", "float32"), ("float16", "half64")])
def test_fast_math_prints_the_fast_functions_and_division(dtype, vector):
    @T.prim_func
    def kernel(A: T.Tensor((4, 256), dtype), B: T.Tensor((4, 256), dtype)):
        with T.Kernel(4) as bx:
            for j in T.Parallel(256):
                B[bx, j] = T.exp(A[bx, j]) / (T.abs(A[bx, j]) + T.cast(1, dtype))

    source = _source(kernel, fast_math=True)
    assert f"tl_hvx_div_fast(tl_hvx_exp_fast(*({vector}*)(A + " in source
    # |x| is exact: no fast variant.
    assert "tl_hvx_fabs(" in source
    accurate = _source(kernel)
    assert "_fast(" not in accurate
    assert f"tl_hvx_div(tl_hvx_exp(*({vector}*)(A + " in accurate


def test_heavy_maps_unroll_by_two():
    @T.prim_func
    def kernel(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((128, 128), "float32")
            T.copy(A, a)
            for i, j in T.Parallel(128, 128):
                a[i, j] = a[i, j] * T.sigmoid(a[i, j])
            for i, j in T.Parallel(128, 128):
                a[i, j] = a[i, j] * a[i, j]
            T.copy(a, B)

    lines = _source(kernel).splitlines()
    # Only the map that computes a math function, two chains an iteration.
    (pragma,) = [i for i, line in enumerate(lines) if "#pragma clang loop unroll_count(2)" in line]
    assert lines[pragma + 1].strip().startswith("for (")
    assert "tl_hvx_sigmoid(" in lines[pragma + 2]


def test_fast_division_by_a_constant_multiplies():
    @T.prim_func
    def kernel(A: T.Tensor((4, 256), "float32"), B: T.Tensor((4, 256), "float32")):
        with T.Kernel(4) as bx:
            for j in T.Parallel(256):
                B[bx, j] = A[bx, j] / T.float32(8)

    source = _source(kernel, fast_math=True)
    assert "tl_hvx_div" not in source
    assert "1.250000e-01f" in source


# ---------------------------------------------------------------------------
# Per-row scalars broadcast from registers (BroadcastFromRegisters).
# ---------------------------------------------------------------------------


def _row_map(rows, cols, row_dtype):
    """S[i, j] = exp(S[i, j] - r[i]) in fp32 over an fp16 tile, as in softmax."""

    @T.prim_func
    def kernel(X: T.Tensor((rows, cols), "float16"), R: T.Tensor((rows,), row_dtype), Y: T.Tensor((rows, cols), "float16")):
        with T.Kernel(1):
            s = T.alloc_shared((rows, cols), "float16")
            r = T.alloc_shared((rows,), row_dtype)
            T.copy(X, s)
            T.copy(R, r)
            for i, j in T.Parallel(rows, cols):
                s[i, j] = T.cast(T.exp(T.cast(s[i, j], "float32") - T.cast(r[i], "float32")), "float16")
            T.copy(s, Y)

    return kernel


def test_row_scalars_come_from_one_register_per_strip():
    source = _source(_row_map(64, 64, "float32"))
    # Strips of 32 rows: one register of fp32 scalars each, broadcast per row.
    assert "for (int32_t i_strip = 0; i_strip < 2; ++i_strip) {" in source
    assert "float32 r_lanes = tl_hvx_load_lanes<32>((&(((float*)r)[(i_strip * 32)])), 32);" in source
    assert "for (int32_t i_lane = 0; i_lane < 32; ++i_lane) {" in source
    assert "tl_hvx_broadcast_lane<32, 1>(r_lanes, i_lane)" in source
    # The row loop reads no scalar of r.
    assert "((float*)r)[" not in source.split("i_strip")[1].split("tl_hvx_load_lanes")[0]


def test_fp16_row_scalars_fill_a_strip_of_64_rows():
    source = _source(_row_map(128, 64, "float16"))
    assert "half64 r_lanes = tl_hvx_load_lanes<64>((&(((half*)r)[(i_strip * 64)])), 64);" in source


def test_the_last_strip_loads_only_its_rows():
    source = _source(_row_map(70, 64, "float32"))
    assert "for (int32_t i_strip = 0; i_strip < 3; ++i_strip) {" in source
    assert "int32_t v_ = 70 - (i_strip * 32);" in source
    assert "tl_hvx_load_lanes<32>((&(((float*)r)[(i_strip * 32)])), ((32) < (v_) ? (32) : (v_)));" in source


def test_rows_of_several_registers_split_into_rows_and_columns():
    # 4096 fp16 columns are 64 registers per row: T.Parallel fuses them into
    # one loop over all the vectors, which splits back into rows of 64.
    source = _source(_row_map(64, 4096, "float32"))
    assert "for (int32_t i_strip = 0; i_strip < 2; ++i_strip) {" in source
    assert "for (int32_t i_lane = 0; i_lane < 32; ++i_lane) {" in source
    assert "for (int32_t i_col = 0; i_col < 64; ++i_col) {" in source
    assert "tl_hvx_broadcast_lane<32, 1>(r_lanes, i_lane)" in source


def test_rows_of_a_few_registers_unroll():
    # 256 fp16 columns are 4 registers per row: the row unrolls, so that the
    # loop over the strip's rows is the innermost.
    source = _source(_row_map(64, 256, "float32"))
    lanes = source.split("for (int32_t i_lane = 0; i_lane < 32; ++i_lane) {")[1].split("tl_crouton")[0]
    assert "i_col" not in source
    assert lanes.count("tl_hvx_broadcast_lane<32, 1>(r_lanes, i_lane)") == 4


def test_scalars_that_do_not_step_with_the_row_stay_loads():
    @T.prim_func
    def kernel(X: T.Tensor((64, 64), "float32"), R: T.Tensor((128,), "float32"), Y: T.Tensor((64, 64), "float32")):
        with T.Kernel(1):
            s = T.alloc_shared((64, 64), "float32")
            r = T.alloc_shared((128,), "float32")
            T.copy(X, s)
            T.copy(R, r)
            for i, j in T.Parallel(64, 64):
                s[i, j] = s[i, j] * r[i * 2]
            T.copy(s, Y)

    source = _source(kernel)
    assert "tl_hvx_broadcast_lane" not in source


@pytest.mark.skipif(__import__("os").environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
@pytest.mark.parametrize(
    "rows,cols,row_dtype", [(64, 64, "float32"), (70, 64, "float32"), (128, 64, "float16"), (100, 64, "float16"), (48, 256, "float32")]
)
def test_row_broadcast_maps_match_torch(rows, cols, row_dtype):
    import torch

    kernel = tilelang.compile(_row_map(rows, cols, row_dtype), out_idx=[2], target="hexagon")
    try:
        gen = torch.Generator().manual_seed(3)
        x = (torch.randn(rows, cols, generator=gen) * 0.5).half()
        r = torch.randn(rows, generator=gen).to(getattr(torch, row_dtype))
        y = kernel(x, r)
        expected = torch.exp(x.float() - r.float()[:, None]).half()
        torch.testing.assert_close(y, expected, rtol=2e-3, atol=2e-3)
    finally:
        kernel.adapter.close()


# Accesses off register boundaries (vmemu).


def _shifted_map(M, N, load_shift, store_shift, dtype):
    """y[i, j + store_shift] = 2 x[i, j + load_shift], in VTCM: rows of N +
    shift elements, so most rows start off a register boundary."""

    @T.prim_func
    def kernel(X: T.Tensor((M, N + load_shift), dtype), Y: T.Tensor((M, N + store_shift), dtype)):
        with T.Kernel(1):
            x = T.alloc_shared((M, N + load_shift), dtype)
            y = T.alloc_shared((M, N + store_shift), dtype)
            T.copy(X, x)
            T.clear(y)
            for i, j in T.Parallel(M, N):
                y[i, j + store_shift] = x[i, j + load_shift] * T.cast(2, dtype)
            T.copy(y, Y)

    return kernel


def test_loads_off_register_boundaries_are_unaligned_vector_loads():
    source = _source(_shifted_map(64, 64, 3, 0, "float16"))
    # x's rows are 67 fp16 apart, y's 64: the loads take vmemu, the stores vmem.
    assert re.search(
        r"\*\(half64\*\)\(\(\(half\*\)y\) \+ \(i_\d+ \* 64\)\) = \(tl_hvx_vec_loadu<half64>\(\(&\(\(\(half\*\)x\)\[\(\(i_\d+ \* 67\) \+ 3\)\]\)\)\)",
        source,
    )
    assert "tl_hvx_vec_storeu" not in source


def test_stores_off_register_boundaries_are_unaligned_vector_stores():
    source = _source(_shifted_map(64, 64, 0, 5, "float16"))
    assert re.search(r"tl_hvx_vec_storeu\(\(&\(\(\(half\*\)y\)\[\(\(i_\d+ \* 69\) \+ 5\)\]\)\), \(\*\(half64\*\)\(\(\(half\*\)x\)", source)
    assert "tl_hvx_vec_loadu" not in source


def test_rows_of_odd_length_vectorize():
    @T.prim_func
    def kernel(X: T.Tensor((8, 200), "float16"), Y: T.Tensor((8, 192), "float16")):
        with T.Kernel(1):
            x = T.alloc_shared((8, 200), "float16")
            T.copy(X, x)
            for i, j in T.Parallel(8, 192):
                Y[i, j] = x[i, j] + x[i, j + 8]

    source = _source(kernel)
    # Row i starts 400 bytes after row i - 1; Y's rows are whole registers.
    (line,) = [line.strip() for line in source.splitlines() if "tl_hvx_vec_loadu" in line]
    assert line.startswith("*(half64*)(Y + (i_1 * 64)) = (tl_hvx_vec_loadu<half64>((&(x[(((i_1 / 3) * 200) + ((i_1 % 3) * 64))])))")
    assert "tl_hvx_vec_loadu<half64>((&(x[((((i_1 / 3) * 200) + ((i_1 % 3) * 64)) + 8)])))" in line


@pytest.mark.skipif(__import__("os").environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
@pytest.mark.parametrize("load_shift, store_shift", [(3, 0), (0, 5), (7, 1), (64, 64)])
@pytest.mark.parametrize("dtype", ["float16", "float32", "int8"])
def test_shifted_maps_match_torch(dtype, load_shift, store_shift):
    import torch

    M, N = 32, 128
    kernel = tilelang.compile(_shifted_map(M, N, load_shift, store_shift, dtype), out_idx=[1], target="hexagon")
    try:
        torch_dtype = getattr(torch, dtype)
        if dtype == "int8":
            x = torch.randint(-50, 50, (M, N + load_shift), dtype=torch_dtype)
        else:
            x = torch.randn(M, N + load_shift).to(torch_dtype)
        expected = torch.zeros(M, N + store_shift, dtype=torch_dtype)
        expected[:, store_shift:] = x[:, load_shift:] * 2
        torch.testing.assert_close(kernel(x), expected, rtol=0, atol=0)
    finally:
        kernel.adapter.close()


def _table_rows(rows, positions):
    """Y[i, d] = X[i, d] * table[base + i, d % 32]: rows of a table at a run-time
    base, which the kernel cannot prove in bounds, as RoPE reads its tables."""

    @T.prim_func
    def kernel(
        table: T.Tensor((positions, 32), "float32"), X: T.Tensor((rows, 64), "float32"), Y: T.Tensor((rows, 64), "float32"), base: T.int32
    ):
        with T.Kernel(1):
            x = T.alloc_shared((rows, 64), "float32")
            y = T.alloc_shared((rows, 64), "float32")
            T.copy(X, x)
            for i, d in T.Parallel(rows, 64):
                y[i, d] = x[i, d] * table[base + i, d % 32]
            T.copy(y, Y)

    return kernel


def test_guarded_vector_loads_select_instead_of_branching():
    source = _source(_table_rows(128, 2048))
    # The guard reads from the table's start when it fails, and selects: no
    # branch around the load.
    (load,) = [line.strip() for line in source.splitlines() if "guarded = " in line]
    assert load.startswith("float32 guarded = *(float32*)(table + (((0 <= ((i_1 >> 1) + base)) && (((i_1 >> 1) + base) < 2048)) ? ")
    assert load.endswith(": (int64_t)0));")
    assert " ? guarded : " in source
    assert "condval" not in source


@pytest.mark.skipif(__import__("os").environ.get("TL_HEXAGON_DEVICE_TEST") != "1", reason="requires an authorized Hexagon device and SDK")
def test_guarded_loads_read_zeros_out_of_bounds():
    import torch

    kernel = tilelang.compile(_table_rows(128, 256), out_idx=[2], target="hexagon")
    gen = torch.Generator().manual_seed(5)
    table, x = torch.randn(256, 32, generator=gen), torch.randn(128, 64, generator=gen)
    # Rows from 200 on: the last 72 are past the table's end, and read zeros.
    out = kernel(table, x, 200)
    ref = x * torch.cat([table, torch.zeros(72, 32)])[200:328].repeat(1, 2)
    torch.testing.assert_close(out, ref)
