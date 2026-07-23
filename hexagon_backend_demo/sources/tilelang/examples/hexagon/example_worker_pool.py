"""Worker pool over Hexagon's 1 HMX + 6 HVX units, via `T.Kernel(num_workers=N)`.

`num_workers=N` fans the grid's outermost block loop across the (up to 6) HW threads.
The 6 HVX units run pack/unpack/elementwise in parallel; the single HMX matrix engine
is serialized by an accumulator spinlock — so the same multi-block kernel parallelizes
whether it's pure-HVX or uses `T.gemm`.

Two demos:
  1. Batched matmul — each block does a `T.gemm` (HMX), parallelized across workers.
     Each worker gets a private VTCM slice (operands + gemm scratch), so concurrent
     gemms don't collide. Shows correctness.
  2. A compute-heavy HVX kernel — timed at num_workers=1 vs N to show the wall-clock
     speedup (this kernel is compute-bound, so the speedup is visible end-to-end;
     for HMX-bound or tiny kernels the win is smaller / USB-hidden).

Requires a Hexagon device + SDK over adb (see README.md).

    python example_worker_pool.py
"""
import time
import torch
import tilelang
import tilelang.language as T

NW = 6  # workers (== the device's HW-thread count)


# ----- demo 1: batched matmul, each block a T.gemm parallelized across workers -----
def make_batched_matmul(NB, M, N, K, num_workers):
    @T.prim_func
    def batched(A: T.Tensor((NB, M, K), "float16"), B: T.Tensor((NB, K, N), "float16"),
                C: T.Tensor((NB, M, N), "float16")):
        with T.Kernel(NB, threads=1, num_workers=num_workers) as bx:
            A_sh = T.alloc_shared((M, K), "float16")   # per-worker VTCM slice
            B_sh = T.alloc_shared((K, N), "float16")
            C_sh = T.alloc_shared((M, N), "float16")
            T.copy(A[bx, :, :], A_sh)
            T.copy(B[bx, :, :], B_sh)
            T.gemm(A_sh, B_sh, C_sh, clear_accum=True)  # HMX (per-worker scratch)
            T.copy(C_sh, C[bx, :, :])
    return batched


# ----- demo 2: a compute-heavy HVX kernel, to show the wall-clock speedup -----
def make_heavy(NB, BS, ITERS, num_workers):
    @T.prim_func
    def heavy(A: T.Tensor((NB * BS,), "float16"), C: T.Tensor((NB * BS,), "float16")):
        with T.Kernel(NB, threads=1, num_workers=num_workers) as bx:
            for i in T.serial(BS):
                acc = T.alloc_local((1,), "float16")
                acc[0] = A[bx * BS + i]
                for _ in T.serial(ITERS):
                    acc[0] = acc[0] * T.float16(0.999)
                C[bx * BS + i] = acc[0]
    return heavy


def demo_batched_matmul():
    NB, M, N, K = NW, 128, 128, 128
    kernel = tilelang.compile(make_batched_matmul(NB, M, N, K, NW), out_idx=[2], target="hexagon")
    a = (torch.randn(NB, M, K) * 0.25).half()
    b = (torch.randn(NB, K, N) * 0.25).half()
    c = kernel(a, b).float()
    err = (c - torch.bmm(a.float(), b.float())).abs().max().item()
    print(f"[1] batched matmul ({NB}x {M}x{N}x{K}) across {NW} workers: "
          f"max abs err = {err:.4g}  ({'PASS' if err < 0.05 else 'FAIL'})")


def demo_speedup():
    NB, BS, ITERS = 48, 256, 3000
    a = (torch.randn(NB * BS) * 0.5).half()
    times = {}
    for nw in (1, NW):
        kernel = tilelang.compile(make_heavy(NB, BS, ITERS, nw), out_idx=[1], target="hexagon")
        kernel(a)  # warm up (build + deploy + spawn the persistent pool)
        ts = []
        for _ in range(5):
            t0 = time.time()
            kernel(a)
            ts.append(time.time() - t0)
        times[nw] = min(ts)
    sp = times[1] / times[NW]
    print(f"[2] compute-heavy HVX grid: num_workers=1 {times[1]*1e3:.0f} ms  "
          f"num_workers={NW} {times[NW]*1e3:.0f} ms  ->  {sp:.2f}x")


if __name__ == "__main__":
    demo_batched_matmul()
    demo_speedup()
