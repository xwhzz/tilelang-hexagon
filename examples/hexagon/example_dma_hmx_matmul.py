"""TileLang HMX matmul with DMA prefetch across output blocks (BK=K).

Two row-major staging slots hold the next output blocks' A/B inputs. After
packing the current slot into separate Crouton buffers, submit block i+2
before the current block's HMX computation.
"""
import torch
import tilelang
import tilelang.language as T
from tilelang.hexagon.hmx_intrin import HMXIntrinEmitter


def make_dma_hmx_matmul(M=256, N=512, K=2048, BM=128, BN=128):
    assert M % BM == N % BN == K % 32 == BM % 32 == BN % 32 == 0
    E = HMXIntrinEmitter(BM, BN, K)
    count = (M // BM) * (N // BN)

    @T.macro
    def submit(A, B, ar, br, tile):
        mi = tile // (N // BN)
        ni = tile % (N // BN)
        T.dma_copy(A[mi * BM:(mi + 1) * BM, 0:K], ar[tile % 2, 0:BM, 0:K])
        T.dma_copy(B[0:K, ni * BN:(ni + 1) * BN], br[tile % 2, 0:K, 0:BN])

    @T.prim_func
    def dma_hmx_matmul(A: T.Tensor((M, K), 'float16'),
                       B: T.Tensor((K, N), 'float16'),
                       C: T.Tensor((M, N), 'float16')):
        T.func_attr({'hexagon.dma_queue_capacity': 4})
        with T.Kernel(1, threads=1):
            ar = T.alloc_shared((2, BM, K), 'float16', align=2048)
            br = T.alloc_shared((2, K, BN), 'float16', align=2048)
            an = T.alloc_shared((BM, K), 'float16', align=2048)
            bn = T.alloc_shared((K, BN), 'float16', align=2048)
            cn = T.alloc_shared((BM, BN), 'float16', align=2048)
            config = T.alloc_shared((64,), 'uint32', align=256)
            acc = T.alloc_hmx_accumulator()
            cvt = T.alloc_hmx_convert_state()
            bias = T.alloc_hmx_bias_state()
            T.annotate_layout({an: E.activation_layout(an), bn: E.weight_layout(bn),
                               cn: E.output_layout(cn)})
            for i in T.serial(32):
                config[i] = T.uint32(0x3C00)
                config[32 + i] = T.uint32(0)
            # Each output block contributes two queue entries: A and B.
            for tile in T.serial(min(2, count)):
                submit(A, B, ar, br, tile)
            for tile in T.serial(count):
                if tile < count - 1:
                    T.dma_wait(2)
                else:
                    T.dma_wait()
                T.copy(ar[tile % 2, 0, 0], an)
                T.copy(br[tile % 2, 0, 0], bn)
                # Packing has finished reading this slot. Refill it while HMX
                # computes from the separate native Crouton buffers.
                if tile + 2 < count:
                    submit(A, B, ar, br, tile + 2)
                E.acquire(acc)
                E.load_bias(bias, config)
                for mt in T.serial(BM // 32):
                    for nt in T.serial(BN // 32):
                        E.clear(acc)
                        for kt in T.serial(K // 32):
                            E.mma_atom(acc, an, bn, mt, nt, kt)
                        E.convert(cvt, acc, bias, config)
                        E.store(cvt, acc, cn, bias, config, mt, nt)
                E.release(acc)
                T.copy(cn, C[(tile // (N // BN)) * BM:((tile // (N // BN)) + 1) * BM,
                             (tile % (N // BN)) * BN:((tile % (N // BN)) + 1) * BN])
    return dma_hmx_matmul


def main():
    M, N, K = 256, 512, 2048
    kernel = tilelang.compile(make_dma_hmx_matmul(M, N, K), out_idx=[2], target='hexagon')
    worst = 0.0
    try:
        for seed in range(5):
            torch.manual_seed(seed)
            a = (torch.randn(M, K) * 0.125).half()
            b = (torch.randn(K, N) * 0.125).half()
            c = kernel(a, b).float()
            ref = a.float() @ b.float()
            assert torch.isfinite(c).all()
            assert torch.allclose(c, ref, atol=0.02, rtol=0.01)
            worst = max(worst, (c - ref).abs().max().item())
        print(f'DMA output-block prefetch + HMX {M}x{N}x{K}: 5/5 PASS, max abs err = {worst:.9g}')
    finally:
        kernel.adapter.close()


if __name__ == '__main__':
    main()
