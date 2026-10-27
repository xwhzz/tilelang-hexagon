"""Five standalone DSL operators for same-interface Hexagon comparisons.

No custom C++ operator bodies are injected. Pointwise work, RMSNorm, casts,
packing schedules and DMA scheduling are expressed in TileLang; existing
copy/reduce/HMX instruction primitives are selected by the backend.
"""

from tilelang.hexagon import language as T
from tvm import tirx
from tilelang.hexagon.intrinsics.hmx import HMXIntrinEmitter
from tilelang.hexagon.language import macro
from tilelang.utils.language import to_tile_region


@macro
def hexreduce(src, dst, rtype, dim):
    """Reduce shared FP16/FP32 rows into FP32, without fragment staging.

    Like example_rmsnorm.py, keep this adapter local until T.reduce_sum/max
    can lower shared tiles directly on Hexagon. Supports sum/max with clear=True;
    NaN propagation is unsupported.
    """
    tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.reduce"),
        to_tile_region(src, "r"),
        to_tile_region(dst, "w"),
        rtype,
        dim,
        True,
    )


def make_elementwise(M, N, kind="add", num_workers=6, scale=0.75, bias=0.125):
    """Contiguous FP32 map; X/Y/O must be distinct for the benchmark contract."""
    if M <= 0 or N <= 0 or kind not in ("add", "mul", "scale"):
        raise ValueError("positive M/N and kind=add, mul or scale required")
    if not 1 <= num_workers <= 6:
        raise ValueError("num_workers must be in [1, 6]")

    if kind == "scale":

        @T.prim_func
        def scale_bias(X: T.Tensor((M, N), "float32"), O: T.Tensor((M, N), "float32")):
            with T.Kernel(M, num_workers=min(M, num_workers)) as row:
                for j in T.Parallel(N):
                    O[row, j] = X[row, j] * T.float32(scale) + T.float32(bias)

        return scale_bias

    @T.prim_func
    def elementwise(X: T.Tensor((M, N), "float32"), Y: T.Tensor((M, N), "float32"), O: T.Tensor((M, N), "float32")):
        with T.Kernel(M, num_workers=min(M, num_workers)) as row:
            for j in T.Parallel(N):
                if kind == "add":
                    O[row, j] = X[row, j] + Y[row, j]
                else:
                    O[row, j] = X[row, j] * Y[row, j]

    return elementwise


def make_rmsnorm_fp32(M, N, eps=1e-5, num_workers=6):
    """FP32 RMSNorm without gamma, matching ggml RMS_NORM semantics."""
    if M <= 0 or N <= 0 or eps <= 0:
        raise ValueError("positive M, N and eps required")
    if not 1 <= num_workers <= 6:
        raise ValueError("num_workers must be in [1, 6]")

    @T.prim_func
    def rmsnorm(X: T.Tensor((M, N), "float32"), O: T.Tensor((M, N), "float32")):
        with T.Kernel(M, num_workers=min(M, num_workers)) as row:
            squares = T.alloc_shared((1, N), "float32", align=128)
            total = T.alloc_shared((1,), "float32")
            inv = T.alloc_local((1,), "float32")
            for j in T.Parallel(N):
                squares[0, j] = X[row, j] * X[row, j]
            hexreduce(squares, total, "sum", 1)
            inv[0] = T.rsqrt(total[0] / T.float32(N) + T.float32(eps))
            for j in T.Parallel(N):
                O[row, j] = X[row, j] * inv[0]

    return rmsnorm


def make_map_tiled(M, N, kind="add", tile_rows=4, group_tiles=4, num_workers=6, dma=True, dma_store=False):
    """Contiguous groups, with optional double-buffered DMA input staging.

    Full static tiles are required; callers retain make_elementwise for tails.
    The group grid may exceed the worker count. Each worker owns its two input
    slots and output tile; no cross-worker synchronization is needed.
    """
    if kind not in ("add", "mul", "scale", "rmsnorm"):
        raise ValueError("unsupported map")
    if min(M, N, tile_rows, group_tiles) <= 0 or M % (tile_rows * group_tiles):
        raise ValueError("M must be divisible by tile_rows * group_tiles")
    if N % 32 or not 1 <= num_workers <= 6:
        raise ValueError("N must be HVX aligned; workers in [1, 6]")
    binary = kind in ("add", "mul")
    groups = M // (tile_rows * group_tiles)
    pending = 2 if binary else 1
    partial_lanes = 128 if N % 128 == 0 else 32

    @T.macro
    def submit(X, Y, sx, sy, row, slot):
        T.dma_copy(X[row : row + tile_rows, :], sx[slot, :, :])
        if binary:
            T.dma_copy(Y[row : row + tile_rows, :], sy[slot, :, :])

    @T.macro
    def compute(sx, sy, sums, inv, dst, out_row, slot):
        for r in T.serial(tile_rows):
            if kind == "rmsnorm":
                inv[0] = T.rsqrt(sums[r] / T.float32(N) + T.float32(1e-5))
            for j in T.Parallel(N):
                if kind == "add":
                    dst[out_row + r, j] = sx[slot, r, j] + sy[slot, r, j]
                elif kind == "mul":
                    dst[out_row + r, j] = sx[slot, r, j] * sy[slot, r, j]
                elif kind == "scale":
                    dst[out_row + r, j] = sx[slot, r, j] * T.float32(0.75) + T.float32(0.125)
                else:
                    dst[out_row + r, j] = sx[slot, r, j] * inv[0]

    @T.prim_func
    def map_tiled(X: T.Tensor((M, N), "float32"), Y: T.Tensor((M, N), "float32"), O: T.Tensor((M, N), "float32")):
        with T.Kernel(groups, num_workers=num_workers) as group:
            sx = T.alloc_shared((2, tile_rows, N), "float32")
            sy = T.alloc_shared((2, tile_rows, N), "float32")
            result = T.alloc_shared((tile_rows, N), "float32")
            partial = T.alloc_shared((1, partial_lanes), "float32")
            vacc = T.alloc_local((partial_lanes // 32,), "float32x32")
            row_sum = T.alloc_local((1,), "float32")
            sums = T.alloc_shared((tile_rows,), "float32")
            inv = T.alloc_local((1,), "float32")
            if dma:
                for i in T.serial(min(2, group_tiles)):
                    submit(X, Y, sx, sy, (group * group_tiles + i) * tile_rows, i)
            for i in T.serial(group_tiles):
                row = (group * group_tiles + i) * tile_rows
                if dma:
                    if i < group_tiles - 1:
                        T.dma_wait(pending)
                    else:
                        T.dma_wait()
                else:
                    T.copy(X[row : row + tile_rows, :], sx[i % 2, :, :])
                    if binary:
                        T.copy(Y[row : row + tile_rows, :], sy[i % 2, :, :])
                if kind == "rmsnorm":
                    for r in T.serial(tile_rows):
                        # Keep independent sums in vector locals. Only spill
                        # the final partials for the existing row-reduce op.
                        for u in T.unroll(partial_lanes // 32):
                            vacc[u] = T.Broadcast(T.float32(0), 32)
                        for k in T.serial(N // partial_lanes):
                            for u in T.unroll(partial_lanes // 32):
                                x = sx[i % 2, r, T.Ramp(k * partial_lanes + u * 32, 1, 32)]
                                vacc[u] = vacc[u] + x * x
                        for u in T.unroll(partial_lanes // 32):
                            partial[0, T.Ramp(u * 32, 1, 32)] = vacc[u]
                        hexreduce(partial, row_sum, "sum", 1)
                        sums[r] = row_sum[0]
                if dma_store:
                    compute(sx, sy, sums, inv, result, 0, i % 2)
                    T.dma_copy(result, O[row : row + tile_rows, :])
                    T.dma_wait()
                else:
                    compute(sx, sy, sums, inv, O, row, i % 2)
                # The current input slot is dead once the result is computed.
                if dma and i + 2 < group_tiles:
                    submit(X, Y, sx, sy, (group * group_tiles + i + 2) * tile_rows, i % 2)

    return map_tiled


def make_gemm_f16_io(M, N, K, BN=128, copy_workers=1, fused_copy=False):
    """FP32 A[M,K], FP16 W[N,K] -> FP32 C[M,N], with FP16 HMX arithmetic.

    All per-call conversion/packing is included. Narrow W blocks use two raw
    DMA slots; blocks of at least 512 rows use one to bound the VTCM footprint.
    After packing, refill the raw slot while HMX reads the separate native
    buffer. K is kept whole to avoid changing accumulation semantics. Shapes
    are bounded by the VTCM grant (oversized live sets fail the allocation guard).
    """
    if min(M, N, K, BN) <= 0 or any(v % 32 for v in (M, N, K, BN)) or N % BN:
        raise ValueError("positive 32-multiple dimensions with N divisible by BN required")
    if not 1 <= copy_workers <= 6:
        raise ValueError("copy_workers must be in [1, 6]")
    copy_annotations = {"hexagon.copy_workers": copy_workers} if copy_workers > 1 else {}
    output_annotations = copy_annotations if M >= 128 else {}
    emitter = HMXIntrinEmitter(M, BN, K, b_transposed=True)
    count = N // BN
    slots = 1 if BN >= 512 else min(2, count)

    @T.prim_func
    def gemm(A: T.Tensor((M, K), "float32"), W: T.Tensor((N, K), "float16"), C: T.Tensor((M, N), "float32")):
        with T.Kernel(1):
            a_dma = T.alloc_shared((M, K), "float32", align=2048)
            a_row = T.alloc_shared((M, K), "float16", align=2048)
            a_native = T.alloc_shared((M, K), "float16", align=2048)
            w_dma = T.alloc_shared((slots, BN, K), "float16", align=2048)
            w_native = T.alloc_shared((BN, K), "float16", align=2048)
            c_native = T.alloc_shared((M, BN), "float16", align=2048)
            c_row = T.alloc_shared((M, BN), "float16", align=2048)
            config = T.alloc_shared((64,), "uint32", align=256)
            acc = T.hmx.alloc_accumulator()
            cvt = T.hmx.alloc_convert_state()
            bias_state = T.hmx.alloc_bias_state()
            T.annotate_layout(
                {
                    a_native: emitter.activation_layout(a_native),
                    w_native: emitter.weight_layout(w_native),
                    c_native: emitter.output_layout(c_native),
                }
            )
            for i in T.serial(32):
                config[i] = T.uint32(0x3C00)
                config[i + 32] = T.uint32(0)
            T.dma_copy(A, a_dma)
            for ni in T.serial(slots):
                T.dma_copy(W[ni * BN : (ni + 1) * BN, :], w_dma[ni, :, :])
            T.dma_wait(slots)
            if fused_copy:
                T.copy(a_dma, a_native, annotations=copy_annotations)
            else:
                for i, j in T.Parallel(M, K):
                    a_row[i, j] = T.cast(a_dma[i, j], "float16")
                T.copy(a_row, a_native, annotations=copy_annotations)
            emitter.acquire(acc)
            emitter.load_bias(bias_state, config)
            for ni in T.serial(count):
                if ni < count - 1:
                    T.dma_wait(slots - 1)
                else:
                    T.dma_wait()
                T.copy(w_dma[ni % slots, :, :], w_native, annotations=copy_annotations)
                if ni + slots < count:
                    T.dma_copy(W[(ni + slots) * BN : (ni + slots + 1) * BN, :], w_dma[ni % slots, :, :])
                for mt in T.serial(M // 32):
                    for nt in T.serial(BN // 32):
                        emitter.clear(acc)
                        emitter.mma_tile(acc, a_native, w_native, mt, nt)
                        emitter.convert(cvt, acc, bias_state, config)
                        emitter.store(cvt, acc, c_native, bias_state, config, mt, nt)
                if fused_copy:
                    T.copy(c_native, C[:, ni * BN : (ni + 1) * BN], annotations=output_annotations)
                else:
                    T.copy(c_native, c_row)
                    for i, j in T.Parallel(M, BN):
                        C[i, ni * BN + j] = T.cast(c_row[i, j], "float32")
            emitter.release(acc)

    return gemm
