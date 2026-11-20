"""Two tasks on two cDSPs at once, and a third that depends on both.

An SA8650 has two cDSPs, FastRPC domains 3 and 4. A Session runs on one of them
(`Session(..., domain=d)`); sessions on both run kernels at the same time when
called from two threads. Stage 1 multiplies X1 = A1 @ B1 on the first cDSP and
X2 = A2 @ B2 on the second, concurrently. Stage 2 waits for both and computes
Y = X1 + X2 on the first cDSP. The cDSPs share no VTCM, addresses or caches, so
X2 reaches the first cDSP through the host.

Requires a Hexagon device (adb or SSH) and the SDK (see README.md). On a device
with one cDSP, `--domains 3 3` runs both sessions on it, taking turns.

    python example_dual_cdsp.py --domains 3 4
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import time

import torch

from example_gemm import make_gemm
from tilelang.hexagon import language as T
from tilelang.hexagon.runtime.session import Session


def make_add(M, N):
    @T.prim_func
    def add(X1: T.Tensor((M, N), "float16"), X2: T.Tensor((M, N), "float16"), Y: T.Tensor((M, N), "float16")):
        with T.Kernel(1):
            for i, j in T.Parallel(M, N):
                Y[i, j] = X1[i, j] + X2[i, j]

    return add


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domains", type=int, nargs=2, default=[3, 4], help="the FastRPC domains of the two cDSPs")
    p.add_argument("--size", type=int, default=1024, help="M = N = K of each GEMM (a multiple of 128)")
    p.add_argument("--repeat", type=int, default=9)
    args = p.parse_args()
    n = args.size

    gemm = (make_gemm(n, n, n, 128, 128, 256), [2])
    nbytes = n * n * 2
    first = Session([gemm, (make_add(n, n), [2])], domain=args.domains[0])
    second = Session([gemm], domain=args.domains[1])
    with first, second:
        # Inputs stay resident on the cDSP that uses them; each call passes buffers.
        a1, b1, a2, b2 = ((torch.randn(n, n) * 0.1).half() for _ in range(4))
        buf1 = [first.buffer(nbytes) for _ in range(5)]  # A1, B1, X1, X2, Y
        buf2 = [second.buffer(nbytes) for _ in range(3)]  # A2, B2, X2
        for buf, x in zip(buf1[:2] + buf2[:2], (a1, b1, a2, b2)):
            buf.write(x)
        stage1 = [lambda: first.kernels[0](*buf1[:3]), lambda: second.kernels[0](*buf2)]

        def timed(run):
            times = []
            for _ in range(args.repeat):
                t = time.perf_counter()
                run()
                times.append(time.perf_counter() - t)
            return sorted(times)[len(times) // 2]

        one = timed(lambda: [task() for task in stage1])
        with ThreadPoolExecutor(2) as pool:
            both = timed(lambda: [f.result() for f in [pool.submit(task) for task in stage1]])
        # Stage 2 depends on both results: X2 crosses to the first cDSP via the host.
        buf1[3].write(buf2[2].read())
        first.kernels[1](*buf1[2:])
        y = torch.frombuffer(bytearray(buf1[4].read()), dtype=torch.float16).reshape(n, n).float()

    ref = a1.float() @ b1.float() + a2.float() @ b2.float()
    err = ((y - ref).abs().max() / ref.abs().max()).item()
    print(f"stage 1, two {n}^3 GEMMs: {one * 1e3:.1f} ms one after the other, {both * 1e3:.1f} ms at once on domains {args.domains}")
    print(f"stage 2, Y = X1 + X2: rel err = {err:.4g}  ({'PASS' if err < 1e-2 else 'FAIL'})")


if __name__ == "__main__":
    main()
