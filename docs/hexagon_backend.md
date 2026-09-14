# Adapting tilelang to a Qualcomm Hexagon NPU backend

This document explains how the `hexagon` backend was added to tilelang, and — more
importantly — *the principles* behind it, so you can read the code (and extend it)
without re-deriving the design.

Target hardware: Qualcomm Hexagon v79 (Snapdragon 8 Gen 4, e.g. OnePlus 13), reached
from an x86 host over `adb` + FastRPC. The backend is device-validated end to end.

---

## 1. The problem: a GPGPU framework meets a VLIW NPU

tilelang is a tile-level DSL designed for **SIMT GPUs**. Its whole mental model —
`T.Kernel` grids of blocks, `threadIdx`-cooperative tiles, shared memory, warp-level
TensorCore `T.gemm` — assumes CUDA-shaped hardware. Hexagon is a different machine:

| | GPGPU (what tilelang targets) | Hexagon NPU (what we map onto) |
|---|---|---|
| Parallelism | 1000s of SIMT threads | VLIW + **HVX** (1024-bit vector) + **HMX** (matrix), ~**6 HW threads** |
| Thread model | implicit `blockIdx`/`threadIdx` | explicit C loops; multithread via a worker pool |
| On-chip memory | per-block shared memory | **VTCM** ~8 MB per core (HAP-acquired arena) |
| Matrix engine | warp-cooperative TensorCores | **standalone HMX coprocessor** — one unit, lock-protected, mandatory *Crouton* tile layout, fp32 accumulator that **can't be preloaded** and outputs **fp16 only** |
| Launch | in-process async | **out-of-process FastRPC** to a separate OS/chip, reached over `adb` from x86 |

The genuinely new engineering is the **out-of-process device** (3-tier: x86 Python ⇄
adb ⇄ Android ARM stub ⇄ Hexagon DSP) and the **standalone HMX** with its Crouton
layout and un-preloadable accumulator. Everything else is "C codegen + a runtime."

---

## 2. Guiding principles

These are the decisions that keep the adaptation small and the code navigable.

1. **Don't touch the DSL.** A Hexagon matmul is written with the *same* `T.Kernel` /
   `T.copy` / `T.gemm` a CUDA user writes. The user declares intent; the backend
   decides hardware. (The one Hexagon-specific knob is `T.Kernel(num_workers=N)`,
   propagated like `cluster_dims`.)

2. **The codegen is a thin C emitter; instruction details live in C templates.**
   `CodeGenTileLangHexagon` emits plain C and maps HMX TIR intrinsics to wrappers in
   `src/tl_templates/hexagon/hmx.h`. `GemmHMX` owns layout inference and composes the
   explicit clear/MAC/convert/store sequence; codegen does not construct that protocol.

3. **Hexagon is a sibling of the CPU C backend, not a GPU codegen.** It emits C +
   HVX/HMX intrinsics, so `CodeGenTileLangHexagon` extends TVM's `CodeGenC` and reuses
   the CPU tile-op lowering, *not* the CUDA codegen. Start every new op from the CPU
   path.

4. **A lean, direct FastRPC runtime.** TVM ships a complete Hexagon runtime, but it's
   compiled out, coupled to TVM's module/minRPC model, and LLVM-only (no HMX). We use
   our own thin direct-FastRPC transport behind a swappable interface and harvest
   TVM's runtime only as reference. No codegen change is needed to swap transports.

5. **Prove on-device early and often.** Every phase ends with a kernel running
   *correctly on the phone* (err vs torch), not just compiling. Design before coding,
   reflect during, run `/code-review max` after each milestone. (This caught the
   silent param-order bug, the VTCM collision, and the worker-0 HMX-enable race.)

---

## 3. The 5-layer architecture

```
 Layer 1  Frontend / TIR ...................... unchanged (shared with all targets)
 Layer 2  hexagon TIR pipeline ............... CPU pipeline + Hexagon tile-op lowering
 Layer 3  CodeGenTileLangHexagon ............. TIR -> C + HVX/HMX intrinsics
 Layer 4  tl_templates/hexagon/*.h ........... ALL hardware knowledge (the recipe)
 Layer 5  tilelang/hexagon/*.py .............. build + deploy + FastRPC + marshal
```

- **Layer 2** — `tilelang/hexagon/pipeline.py` reuses the CPU pass pipeline and
  registers the Hexagon `T.gemm` impls (`GemmHMX` for fp16 ×32 shapes, `GemmScalar`
  fallback). Op lowering for `T.copy`/`T.gemm`/`T.fill` lives in `src/hexagon/op/*.cc`,
  each gated on `TargetIsHexagon` and reusing the CPU op where possible.
- **Layer 3** — `src/hexagon/codegen/codegen_hexagon.{cc,h}`. The key overrides:
  serialize the GPU grid (`VisitStmt_(AttrStmtNode)` — `thread_extent` → nested `for`
  loops); place `alloc_shared` in VTCM (`VisitStmt_(AllocBufferNode)`); emit the
  worker-pool dispatch + per-worker VTCM (`AddFunction`); rewrite a worker-pool
  `T.gemm` to the per-worker variant (`VisitExpr_(CallNode)`).
- **Layer 4** — the recipe headers (Section 6/7). `common.h` (types), `vtcm.h` (the
  arena), `hmx.h` (the HMX matmul + HVX Crouton pack/unpack + session), `worker.h`
  (the HW-thread pool).
- **Layer 5** — `tilelang/hexagon/`: `_fastrpc.py` generates the IDL + skel + host +
  CMake from the kernel params; `build.py`/`env.py` drive the Hexagon SDK build;
  `agent.py` runs a persistent on-DSP agent over a socket (`adb forward`);
  `adapter.py` marshals torch tensors and invokes.

---

## 4. The seams — where you plug into tilelang

A new target is wired in at a handful of dispatch points keyed on `target.kind.name`:

- `tilelang/utils/target.py` — `SUPPORTED_TARGETS` + `TargetIsHexagon`
- `tilelang/jit/execution_backend.py` — allowed backends for the target
- `tilelang/engine/lower.py` — `device_codegen` → `target.build.tilelang_hexagon`
- `tilelang/cache/__init__.py` + `jit/kernel.py` — `_dispatch_map["hexagon"]` →
  `HexagonKernelCache` (in-memory only — a Hexagon "artifact" is a live device
  deployment, not a serializable `.so`)
- `src/hexagon/codegen/rt_mod_hexagon.cc` — registers `target.build.tilelang_hexagon`
- `src/hexagon/op/*.cc` (+ the CMake GLOB) — per-op lowering impls
- `tilelang/hexagon/pipeline.py` — `register_gemm_impl(...)` for the gemm dispatch

Touch a comparable set to add any new backend; grep `kDLHexagon` / `"hexagon"` to see
them all.

---

## 5. The lowering flow (DSL → on-device)

```
@T.prim_func ─► TIR
   │  resolve_pipeline("hexagon").lower()  (LowerOpaqueBlock, LowerTileOp→HMX/HVX, ...)
   │  SplitHostDevice  (device PrimFunc; ORDER params by host decl order — see §7)
   ▼
CodeGenTileLangHexagon  ─►  C + HVX/HMX intrinsics   (the generated kernel)
   +  src/tl_templates/hexagon/*.h                   (the hardware recipe)
   ▼
_fastrpc.write_project  ─►  FastRPC skel (.idl/.c/.cc) + host + CMake
   ▼
hexagon-clang++ (-mhmx -mhvx)  ─►  skel .so   ──adb push──►  cDSP user PD
   ▲                                                              │
   └───────── adapter marshals torch tensors over FastRPC ───────┘
```

You can run it three ways: `tilelang.lower(fn, target="hexagon")` (inspect the C),
`tilelang.compile(...)` / `@tilelang.jit(target="hexagon")` (build+deploy+run),
or the direct `HexagonKernelAdapter`. Runnable end-to-end examples (matmul, flash
attention, worker pool) live in [`examples/hexagon/`](../examples/hexagon/).

---

## 6. How DSL constructs map to Hexagon

| DSL | Generated C | Recipe (Layer 4) |
|---|---|---|
| `T.Kernel(grid)` | nested `for` loops (grid serialized on one HW thread) | — |
| `T.Kernel(..., num_workers=N)` | outermost `blockIdx.x` loop **strided across HW threads** via `tl_parallel` | `worker.h` |
| `T.alloc_shared` | a byte offset into the VTCM arena `tl_vtcm_base()+off` | `vtcm.h` |
| `T.copy(global, hmx_shared)` | strided DDR -> native Crouton pack (`vshuff` on the common path) | Hexagon copy/layout lowering |
| `T.copy(hmx_shared, global)` | native Crouton -> strided DDR unpack (`vdeal` on the common path) | Hexagon copy/layout lowering |
| `T.copy(row_shared, hmx_shared)` | native Crouton pack/unpack | Hexagon copy/layout lowering |
| `T.gemm(A,B,C)` | explicit HMX atom loops; each result stores directly into its 2 KB-aligned native C tile | `gemm_hmx.py`, `hmx.h` |

Reading generated C for a tiled matmul now shows the grid loops, strided
`tl_hexagon_hmx_pack_crouton`/`unpack_crouton` calls at the DDR/native boundary,
native Crouton VTCM offsets, and explicit HMX atoms.

---

## 7. The hard-won lessons (gotchas worth knowing before you touch the recipe)

- **Crouton layout.** HMX reads operands from VTCM in a mandatory 32×32-tile layout:
  `cpos(i,j) = (i&~1)*32 + j*2 + (i&1)` — a **row-pair interleave**. That interleave
  is exactly one HVX `Q6_W_vshuff_VVR` of two source rows per 64-column span (and
  `vdeal` to unpack). `GemmHMX.infer_layout` assigns this native layout directly. A full
  FP16 VTCM-to-VTCM `T.copy` with exactly one HMX-layout endpoint is recognized by the
  Hexagon copy lowering and uses those helpers. Widths not divisible by 64 and transposed
  source storage retain scalar fallbacks. The row-major stride is explicit in the helper
  ABI. DMA remains a separate row-major staging operation: the DMA descriptor can move
  contiguous/row-strided rectangles but cannot express the Crouton permutation.

- **DMA primitives are invoked explicitly.** `dma.h` provides raw User-DMA
  instruction wrappers, 1D/type-9 2D descriptor construction, cache maintenance,
  and a caller-owned descriptor ring with start/link/poll/wait/pop/flush operations.
  The queue allocates nothing and does not acquire or reset DMA0; its caller must
  own and serialize the engine. Synchronous 1D/2D copy helpers remain available
  through explicit extern calls. `T.dma_copy` submits to a per-kernel FIFO and
  `T.dma_wait(n)` completes/reclaims all but its newest n entries (one copy per
  entry). The kernel drains outstanding transfers before returning. Capacity is
  configured with `hexagon.dma_queue_capacity` (default 16, range 1..256); a full
  queue fails without a hidden wait. Regions must have equal scalar dtypes, static
  positive extents/strides, unit innermost stride, and provable bounds. Unsupported
  async copies fail lowering. Managed async kernels disable automatic VTCM
  allocation reuse and require a manual serial schedule and exclusive DMA0
  ownership; compiler pipelines and mixing raw DMA calls are rejected. Scope-based
  cleanup drains DMA and releases an acquired HMX atom lock on errors; these DSP
  projects compile with `-fno-exceptions` to avoid unsupported unwinder symbols.
  `T.copy` has no DMA-specific lowering; native
  Crouton endpoints retain HMX pack/unpack. DMA calls inside `num_workers` are
  rejected until engine ownership and queue state are worker-local.

- **The HMX accumulator can't be preloaded and outputs fp16 only.** So **K is not the
  pipeline axis** — you reduce the whole inner-K for an output tile in one `mxclracc`
  sequence; outer-loop / `clear_accum=False` accumulation lives in HVX/VTCM (decompose
  into an overwrite-gemm-to-temp + HVX add — this is how flash attention works).

- **HMX operands have role-specific alignment.** Activation and output need 2048-byte
  VTCM alignment, weight needs 128 bytes, and the scale/bias config block needs 256
  bytes. The shared-memory planner propagates each requirement from the corresponding
  HMX intrinsic operand. Every 32x32 FP16 output tile is itself 2048 bytes, so `T.gemm`
  can advance between native C tiles and store directly without an HVX staging copy.

- **Param-order ABI bug (subtle, was silent).** `SplitHostDevice::SortDeviceParams`
  orders device-kernel params *alphabetically*; correct for GPU (the host wrapper
  reorders args) but the Hexagon FastRPC skel calls the kernel **directly by position**
  in declaration order. Fix: `OrderByHostParams` for `kDLHexagon`. *Lesson: validate
  with non-alphabetical param names + non-square shapes, or symmetry hides the bug.*

- **VTCM is one arena; partition it carefully.** Codegen bump-allocates `alloc_shared`
  bottom-up and *publishes a high-water mark* (`tl_vtcm_shared_high_water`) so the
  HMX gemm's top-down Crouton scratch can't overlap the live tiles. Under the worker
  pool, VTCM is **partitioned per worker** (`base + wid*stride`), and a gemm-in-worker
  carves its scratch from the worker's slice (`region_end`/`op_floor`), not the global
  top.

- **The worker pool = 1 HMX + 6 HVX.** The producer/consumer model: the 6 HVX units
  run pack/unpack/softmax/copies in parallel; the 1 HMX MAC is serialized by a
  `memw_locked` spinlock around the accumulator sequence. Each HW thread enables HMX
  once via `HAP_compute_res_hmx_lock2(SHARED)`, **keyed on thread id** so the session
  thread (which `_open` already enabled) doesn't double-enable. The pool is
  **persistent** (spawned once, blocks on semaphores) so per-call spawn doesn't
  dominate tiny ops. *Never `qurt_hvx_lock`* — just never spawn more workers than HVX
  contexts, and QuRT assigns them implicitly (the htp-ops-lib pattern).

- **fp16 `T.exp` is unresolved** (`FloatSuffix` returns "" for fp16) — compute softmax
  in fp32 (cast → exp → back).

- **Don't import CPU thread-var passes.** Hexagon serializes `thread_extent` into
  loops *at codegen*, keeping the thread var alive; the CPU `ThreadVarCanonicalizer`
  zeroes a (assumed-dead) thread var — applying it to Hexagon corrupts indexing.

---

## 8. Extending the backend (a recipe for the next developer)

- **Add a tile-op** (e.g. a reduction): add `src/hexagon/op/<op>.cc` with a C++ impl
  matching `TargetIsHexagon` (reuse the CPU lowering if it auto-vectorizes onto HVX);
  add it to the CMake GLOB (touch `CMakeLists.txt` to force re-glob).
- **Add a hardware recipe** (e.g. a new HMX kernel): put the C in
  `tl_templates/hexagon/*.h`, call it via `T.call_extern(...)` from an op's `lower`,
  and make `_fastrpc.gen_dsp` `#include` the header when the source mentions its name
  (the lazy-include scheme). No compiler rebuild needed.
- **Make it idiomatic** (a tile-op instead of `call_extern`): register a
  `GemmBase`-style impl + `register_gemm_impl` (see `gemm_hmx.py`), so `T.gemm` lowers
  to it directly.
- **Surface a device-side failure** (VTCM/HMX unavailable, an unmet precondition):
  the generated kernel entry returns an `int32` status — `TL_OK` / `TL_ERR_VTCM` /
  `TL_ERR_HMX` / `TL_ERR_DMA` (`tl_templates/hexagon/common.h`). In a `num_workers` kernel the worker
  callback returns the code and `tl_parallel` OR-reduces them into the entry. The
  FastRPC skel maps any nonzero to `AEE_EFAILED`, so the host's `run()` *raises* rather
  than returning unwritten/partial output (the silent-wrong-output trap). The codegen
  emits the `return`s; a new recipe that can fail should return a nonzero code rather
  than fault or skip silently.
- **Validate** with the `HexagonKernelAdapter` + a small torch reference (the
  `test_*.py` pattern). The persistent agent (`agent.py`) makes the build→deploy→run
  loop ~5 ms instead of ~170 ms per call.

---

## 9. Status and what's left

**Device-validated:** the native-layout emitter-based `T.gemm` path passes the standalone
32×128×128 q4 regression (`rel err = 0.000351`) and the fused-copy 256×256×256 FP16
offline matmul (`max abs err = 0.0009766`) on v79. The earlier monolithic row-major path
also validated idiomatic `T.gemm`→HMX
(~17 TFLOPS), single-block and flash attention, HVX-vectorized data movement,
`@tilelang.jit(target="hexagon")`,
and a full **1-HMX/6-HVX worker pool** that parallelizes any multi-block kernel via
`T.Kernel(num_workers=N)` — including `alloc_shared` blocks (per-worker VTCM) and
`T.gemm` (historically per-worker scratch), with a persistent pool (2–4.5× speedups).
Correctness is fresh; performance of the native-layout replacement has not yet been
benchmarked against the earlier path. Unrecoverable
device conditions (VTCM grant too small, per-worker HMX enable fails) propagate to the
host as a raised error via the int32 kernel-status ABI rather than silent wrong output.

The DMA primitive layer is device-validated on v79 with two linked 1D descriptors and a
strided type-9 2D descriptor in both directions. This establishes descriptor, dmlink,
done-bit reclamation, cache, stride, and error-propagation semantics.
`example_dma_hmx_matmul.py` expresses a two-slot
output-block prefetch schedule in TileLang: submit two blocks, complete the
oldest A/B pair, pack it into separate native Crouton buffers, and submit block
i+2 into the released row-major slot before executing the current block's HMX
MACs. Each output block spans the full K dimension. `benchmark_dma_matmul.py`
compares serial and prefetch versions using DSP-local SDK timing; neither file
imports a handwritten C/C++ compute or DMA scheduling kernel.

**Optional, not yet done:** autotune (`num_workers` + tile sizes as knobs);
DMA engine ownership integration with worker pools; automatic software-pipeline
DMA lowering; and schedules for blocks larger than VTCM. Explicit DSL DMA
submission/wait and DSP timing are available in the examples above.

---

### One-paragraph summary

We adapted a GPGPU tile DSL to a VLIW NPU by **keeping the DSL hardware-agnostic**,
making the **codegen a thin C emitter**, and pushing **all HMX/HVX/VTCM/Crouton
knowledge into C template headers** behind one call per tile-op. Hexagon is treated as
a **C-codegen sibling of the CPU backend** (not a GPU codegen), driven by a **lean
direct-FastRPC runtime**. The result: the same `T.gemm` the user would write for CUDA
runs idiomatically on the HMX matrix engine, composes into flash attention, and
parallelizes across the 6 HW threads with the 1 HMX serialized — all validated on real
hardware.
