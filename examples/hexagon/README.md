# Hexagon backend examples

Runnable tilelang kernels on the Qualcomm Hexagon NPU (HMX matrix engine + HVX vector
units), driven from an x86 host over FastRPC. They use the **public** API
(`tilelang.compile(..., target="hexagon")`) — the same kernels you'd write for a GPU,
lowered onto Hexagon. See [`docs/hexagon_backend.md`](../../docs/hexagon_backend.md)
for how the adaptation works and [`docs/hexagon_dsl_kernels.md`](../../docs/hexagon_dsl_kernels.md)
for what makes a kernel fast.

| Example | Shows |
|---|---|
| `example_gemm.py` | the tiled GEMM with an fp32 accumulator over a pipelined K loop: DMA ahead, HMX within each call, HVX across calls |
| `example_flash_attention.py` | two HMX gemms and an HVX online softmax on their Crouton tiles, composed on-chip |
| `example_ffn.py` | a gated FFN: three HMX gemms and an HVX `silu(h1) * h3` gate on Crouton tiles |
| `example_rmsnorm.py` | HVX `map` + `reduce` (square, rowsum, rsqrt, normalize) |
| `example_gemv_w4a16.py` | a w4a16 decode GEMV: `T.gemm_blockscaled` of int16 activations by 4-bit panel-layout weights on HVX, streamed by DMA across `num_workers` workers |
| [`lfm/`](lfm/) | LFM2-1.2B end to end: GPTQ weights, a decode megakernel that runs every token of a generation in one call, and an HMX prefill of a whole prompt in one call; LFM2 and LFM3-1.7B in fp16, over several DSP processes at long contexts |

## Prerequisites

Follow the environment setup guide to build TileLang with the Hexagon SDK and
CPU PyTorch, and configure either an Android device over ADB or a Linux BSP over
SSH. It includes Docker and local installation, SA8650 firmware settings and a
small board smoke test. See the [setup guide](../../docs/hexagon_env_setup.md).

The example configurations below were developed on a OnePlus 13 (Snapdragon 8
Elite, Hexagon v79). Choose the DSP architecture and example shapes supported by
your target when running on another board.

The first run of each kernel builds and deploys a FastRPC skel and starts a small
persistent on-device agent (kept warm across calls). Build products are cached on
disk under `$TILELANG_CACHE_DIR/hexagon_builds`, keyed by the generated sources,
the Hexagon template headers and the toolchain, so a rerun of an unchanged kernel
only deploys it.

## Running

```bash
cd examples/hexagon

python example_gemm.py --m 256 --n 256 --k 1024 --block-k 256
python example_flash_attention.py --seq 256
python example_ffn.py
python example_rmsnorm.py --m 64 --n 256
python example_gemv_w4a16.py --k 2048 --n 8192
```

Each prints its error against a torch reference and `PASS` or `FAIL`. The LFM example
has its own [README](lfm/README.md).

## Device tests

`TL_HEXAGON_DEVICE_TEST=1` enables the device tests in `testing/python/hexagon`.
Building kernels takes most of their time, so they can run in parallel workers
(pytest-xdist) that share a device lock: an adapter holds the lock from deployment
to `close()`, so one session at a time uses the device while the other workers build.
On a device with two cDSPs, `TL_HEXAGON_DOMAINS=3,4` runs the concurrent-session
test on both; by default it opens both sessions on `TILELANG_HEXAGON_DOMAIN`.

```bash
TL_HEXAGON_DEVICE_TEST=1 TILELANG_HEXAGON_DEVICE_LOCK=/tmp/tl_hexagon_device.lock \
    python -m pytest testing/python/hexagon -n 4
```

## Notes

- **`num_workers`** requests an upper bound. Actual workers are capped by the
  device's reported HW threads, 128-byte HVX contexts, and the SDK's
  `QURT_MAX_HTHREAD_LIMIT` storage capacity (currently 8);
  unavailable queries fall back to the calling thread alone. Each worker has its
  own buffers and VTCM arena. Ordinary grids distribute their programs over the
  available workers. A grid using `T.sync_grid()` must fit entirely or fails with
  `TL_ERR_GRID` before any program starts. Workers help HVX-heavy, streaming and
  batched work; a single HMX-bound matmul is limited by the one matrix engine.
- **Explicit async DMA is `T.dma_copy(src, dst)` and `T.dma_wait(pending=0)`.** Each
  copy is one group of the tile program's queue; a wait completes the older copies
  until at most the newest `pending` are in flight. Wait before reading a destination
  or modifying a source. Every tile program drains its queue before it ends, and the
  compiler sizes the queue for the most copies the program keeps in flight.
  `T.Pipelined` schedules copies the same way on its own.
- If a run reports an agent or VTCM error, close its session and inspect its
  deployment directory and logs. On a shared device, stop only the processes
  owned by that run.
