"""DSP-timed TileLang serial DMA vs two-slot output-block prefetch.

Run with --out DIR. The only external call is the SDK timer; DMA, packing,
HMX and output scheduling are expressed in TileLang. Reports warm resident
input timings, excluding FastRPC round trips.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import tilelang
import tilelang.language as T

from tilelang.hexagon.hmx_intrin import HMXIntrinEmitter


def make_bench(M, N, K, BM, BN):
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
    def dma_matmul_bench(A: T.Tensor((M, K), 'float16'),
                             B: T.Tensor((K, N), 'float16'),
                             control: T.Tensor((2,), 'int32'),
                             C: T.Tensor((M, N), 'float16'),
                             timing: T.Tensor((1,), 'float32')):
        T.func_attr({'hexagon.dma_queue_capacity': 4})
        with T.Kernel(1, threads=1):
            T.import_source('#include <HAP_perf.h>\n')
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
            start = T.call_extern('uint64', 'HAP_perf_get_time_us')
            for repeat in T.serial(control[1]):
                if control[0] != 0:
                    for tile in T.serial(min(2, count)):
                        submit(A, B, ar, br, tile)
                for tile in T.serial(count):
                    if control[0] == 0:
                        submit(A, B, ar, br, tile)
                    if control[0] != 0 and tile < count - 1:
                        T.dma_wait(2)
                    else:
                        T.dma_wait()
                    T.copy(ar[tile % 2, 0, 0], an)
                    T.copy(br[tile % 2, 0, 0], bn)
                    if control[0] != 0 and tile + 2 < count:
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
                T.dma_wait()
            timing[0] = T.Cast('float32', T.call_extern('uint64', 'HAP_perf_get_time_us') - start)
    return dma_matmul_bench


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--configs',default='128,128,256,32,32;256,256,256,64,64;256,256,1024,64,64;256,512,2048,64,64;256,512,2048,64,128;256,512,2048,128,128;128,512,4096,64,128')
    p.add_argument('--samples',type=int,default=15); p.add_argument('--repeats',type=int,default=10)
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True); results=[]
    for dims in args.configs.split(';'):
        M,N,K,BM,BN=map(int,dims.split(','))
        assert M%BM==N%BN==K%32==BM%32==BN%32==0
        print('BUILD',dims,flush=True)
        kernel=tilelang.compile(make_bench(M,N,K,BM,BN),out_idx=[3,4],target='hexagon')
        try:
            (args.out/f'source_{M}_{N}_{K}_{BM}_{BN}.cc').write_text(kernel.get_kernel_source())
            errors=[0.,0.]; same=True
            for seed in range(3):
                torch.manual_seed(seed); a=(torch.randn(M,K)*.125).half(); b=(torch.randn(K,N)*.125).half()
                ref=a.float()@b.float(); outputs=[]
                for mode in (0,1):
                    c,_=kernel(a,b,torch.tensor([mode,1],dtype=torch.int32))
                    assert torch.isfinite(c).all()
                    errors[mode]=max(errors[mode],(c.float()-ref).abs().max().item()); outputs.append(c)
                    assert torch.allclose(c.float(),ref,atol=.02,rtol=.01)
                same &= torch.equal(*outputs)
            assert same
            for _ in range(5):
                for mode in (0,1): kernel(a,b,torch.tensor([mode,args.repeats],dtype=torch.int32))
            times=[[],[]]
            for s in range(args.samples):
                for mode in ((0,1) if s%2==0 else (1,0)):
                    c,t=kernel(a,b,torch.tensor([mode,args.repeats],dtype=torch.int32))
                    assert torch.isfinite(c).all() and torch.allclose(c.float(),ref,atol=.02,rtol=.01)
                    times[mode].append(float(t.item())/args.repeats)
            row=dict(M=M,N=N,K=K,BM=BM,BN=BN,BK=K,repeats=args.repeats,samples=args.samples,
                warmup_batches_per_mode=5,correctness_seeds=3,atol=.02,rtol=.01,
                serial_max_abs=errors[0],prefetch_max_abs=errors[1],bitwise_equal=same,
                serial_us=times[0],prefetch_us=times[1],
                serial_median=float(np.median(times[0])),prefetch_median=float(np.median(times[1])),
                speedup=float(np.median(times[0])/np.median(times[1])))
            results.append(row); (args.out/'dma_results.json').write_text(json.dumps(results,indent=2))
            print('RESULT',json.dumps(row),flush=True)
        finally:
            kernel.adapter.close()
    with (args.out/'dma_summary.csv').open('w') as f:
        keys=[k for k in results[0] if k not in ('serial_us','prefetch_us')]
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(results)

if __name__=='__main__': main()
