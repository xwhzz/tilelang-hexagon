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
| [`lfm2/`](lfm2/) | LFM2-1.2B end to end: GPTQ weights, and a decode megakernel that runs every token of a generation in one call |

## Prerequisites

These examples run on **real hardware** (not an emulator), so they need:

- A Hexagon device — developed/validated on a **OnePlus 13 (Snapdragon 8 Gen 4,
  Hexagon v79)** — connected and **authorized over `adb`** (`adb devices` shows it).
- The **Qualcomm Hexagon SDK** installed, with `HEXAGON_SDK_ROOT` / the SDK env set
  (the build harness compiles a FastRPC skel with `hexagon-clang++ -mhmx -mhvx`); see
  [`docs/hexagon_env_setup.md`](../../docs/hexagon_env_setup.md).
- tilelang built from this repo with the Hexagon backend, and `torch` for the
  reference check.

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

Each prints its error against a torch reference and `PASS` or `FAIL`. The LFM2 example
has its own [README](lfm2/README.md).

## Device tests

`TL_HEXAGON_DEVICE_TEST=1` enables the device tests in `testing/python/hexagon`.
Building kernels takes most of their time, so they can run in parallel workers
(pytest-xdist) that share a device lock: an adapter holds the lock from deployment
to `close()`, so one session at a time uses the device while the other workers build.

```bash
TL_HEXAGON_DEVICE_TEST=1 TILELANG_HEXAGON_DEVICE_LOCK=/tmp/tl_hexagon_device.lock \
    python -m pytest testing/python/hexagon -n 4
```

## Notes

- **`num_workers`** runs the grid's tile programs on up to that many HW threads, each
  worker with its own buffers and VTCM arena. It helps HVX-heavy, streaming and
  batched work; a single HMX-bound matmul is limited by the one matrix engine. Worker
  programs that run together can meet at `T.sync_grid()`, as the LFM2 decode does.
- **Explicit async DMA is `T.dma_copy(src, dst)` and `T.dma_wait(pending=0)`.** Each
  copy is one group of the tile program's queue; a wait completes the older copies
  until at most the newest `pending` are in flight. Wait before reading a destination
  or modifying a source. Every tile program drains its queue before it ends, and the
  compiler sizes the queue for the most copies the program keeps in flight.
  `T.Pipelined` schedules copies the same way on its own.
- If a run reports an agent or VTCM error, clear stale agents: `adb shell pkill -f _agent`.
