# Hexagon backend examples

Runnable tilelang kernels on the Qualcomm Hexagon NPU (HMX matrix engine + HVX vector
units), driven from an x86 host over FastRPC. They use the **public** API
(`tilelang.compile(..., target="hexagon")`) — the same kernels you'd write for a GPU,
lowered onto Hexagon. See [`docs/hexagon_backend.md`](../../docs/hexagon_backend.md)
for how the adaptation works.

| Example | Shows |
|---|---|
| `example_matmul.py` | tiled `T.gemm` → the HMX matrix engine |
| `example_flash_attention.py` | two HMX gemms + an HVX online softmax, composed on-chip |
| `example_worker_pool.py` | `T.Kernel(num_workers=N)` → parallel across the 6 HW threads |

## Prerequisites

These examples run on **real hardware** (not an emulator), so they need:

- A Hexagon device — developed/validated on a **OnePlus 13 (Snapdragon 8 Gen 4,
  Hexagon v79)** — connected and **authorized over `adb`** (`adb devices` shows it).
- The **Qualcomm Hexagon SDK** installed, with `HEXAGON_SDK_ROOT` / the SDK env set
  (the build harness compiles a FastRPC skel with `hexagon-clang++ -mhmx -mhvx`).
- tilelang built from this repo with the Hexagon backend (the `tl` conda env in our
  setup), and `torch` for the reference check.

The first run of each kernel builds + deploys a FastRPC skel and starts a small
persistent on-device agent (kept warm across calls); rebuilds are cached in-process.

## Running

```bash
cd examples/hexagon

python example_matmul.py --m 256 --n 256 --k 256 --block 64
python example_flash_attention.py --seq 256
python example_worker_pool.py
```

## Expected output

```
matmul 256x256x256 (block 64) on HMX: max abs err = 0.001947  (PASS)

flash attention (M=64, SEQ=256, BN=64, D=64) on HMX+HVX: max abs err = 7.492e-05  (PASS)

[1] batched matmul (6x 128x128x128) across 6 workers: max abs err = 0.0009701  (PASS)
[2] compute-heavy HVX grid: num_workers=1 1056 ms  num_workers=6 272 ms  ->  3.89x
```

(Errors are fp16 rounding; the matmul runs the HMX engine at ~17 TFLOPS, and the
worker pool fans the grid across the 6 HW threads with the 1 HMX serialized.)

## Notes

- **`num_workers`** distributes the grid's outermost block loop across HW threads. It
  helps HVX-heavy and batched/multi-block work; a single HMX-bound matmul is limited
  by the one matrix engine (the spinlock serializes the MACs). The compute-heavy demo
  is HVX-bound, so its speedup is visible end-to-end.
- The flash-attention softmax runs on **HVX**: the elementwise `T.serial` maps
  (`exp`, rescale, `/l`) are vectorized to 64-lane HVX by the Hexagon codegen, and the
  row reductions go through the reduce tile op (`hexreduce`). It's spelled with explicit
  `T.serial` loops + a direct reduce call rather than `T.reduce_max`/`T.Parallel`
  because the backend has no single-thread fragment-layout inference — see the kernel
  comments and the adaptation doc.
- If a run reports an agent/VTCM error, clear stale agents: `adb shell pkill -f _agent`.
