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
   which runs the grid's tile programs on the worker pool.)

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
 Layer 2  hexagon TIR pipeline ............... CPU-style pass order + Hexagon tile-op lowering
 Layer 3  CodeGenTileLangHexagon ............. TIR -> C + HVX/HMX intrinsics
 Layer 4  tl_templates/hexagon/*.h ........... ALL hardware knowledge (the recipe)
 Layer 5  tilelang/hexagon/*.py .............. build + deploy + FastRPC + marshal
```

- **Layer 2** — `tilelang/hexagon/pipeline.py` owns the full Hexagon pass order (the
  CPU-style sequence plus VTCM arena planning) and registers the Hexagon `T.gemm`
  impls (`GemmHMX` for fp16 ×32 shapes, `GemmScalar` fallback). Op lowering for `T.copy`/`T.gemm`/`T.fill` lives in `src/hexagon/op/*.cc`,
  each gated on `TargetIsHexagon` and reusing the CPU op where possible. The launch
  grid becomes loops in TIR, as on CPU: `src/hexagon/transform/` marks the grid
  loops (`LowerKernelLaunch`) and, after the loop transformations, sinks each tile
  program's buffers back into its body and turns a `num_workers` grid into one
  parallel loop (`MaterializeGrid`).
- **Layer 3** — `src/hexagon/codegen/codegen_hexagon.{cc,h}`. The key overrides:
  print a worker loop as a `tl_parallel_for` call whose lambda runs one tile program
  (`VisitStmt_(ForNode)`); place `alloc_shared` in the tile program's VTCM arena
  (`VisitStmt_(AllocBufferNode)`).
- **Layer 4** — the recipe headers (Section 6/7). `common.h` (types), `vtcm.h` (the
  arena), `hmx.h` (the HMX atoms + session), `crouton.h` (copies to and from
  Crouton matrices), `worker.h`
  (the HW-thread pool).
- **Layer 5** — `tilelang/hexagon/runtime/`: `fastrpc.py` generates the IDL + skel +
  host + CMake from the kernel params; `build.py`/`env.py` drive the Hexagon SDK
  build; `agent.py` runs a persistent on-DSP agent over a socket (`adb forward`);
  `adapter.py` marshals torch tensors and invokes.

---

## 4. The seams — where you plug into tilelang

The backend registers itself through upstream's backend plugin interface
(`tilelang/backend/README.md`):

- `tilelang/hexagon/backend.py` — the `BackendModule` manifest: pipeline, device
  codegen (`target.build.tilelang_hexagon`) and the `hexagon` execution backend;
  imported from `tilelang/__init__.py`
- `tilelang/hexagon/target.py` — target normalization: the DSP architecture in
  `mcpu`, and the capabilities passes read (`max_vector_bits`, `vtcm-capacity`,
  `num-cores`). `src/hexagon/target.cc` adds `max_vector_bits` (1024) to the TVM
  `hexagon` kind, so the vectorizer needs no Hexagon special case
- `tilelang/hexagon/pipeline.py` — the pass order
- `tilelang/hexagon/op/` — tile-op implementations; `op/gemm/` registers the
  gemm dispatch (`register_gemm_impl(...)`)
- `tilelang/hexagon/intrinsics/` — Crouton layouts and the HMX emitter. The ggml
  Q4_0/Q8_0 format helpers live with the examples in `examples/hexagon/quantize/`
- `tilelang/hexagon/language/` — the Hexagon dialect: `T.Kernel(num_workers=...)`,
  `T.dma_copy`/`T.dma_wait`, and the expert HMX atoms under `T.hmx`. Kernels import
  it as `from tilelang.hexagon import language as T`
- `tilelang/cache/__init__.py` + `jit/kernel.py` — `_dispatch_map["hexagon"]` →
  `HexagonKernelCache` (in-memory only — a Hexagon "artifact" is a live device
  deployment, not a serializable `.so`) and the `HexagonKernelAdapter`
- `src/hexagon/CMakeLists.txt` — native sources; `src/hexagon/target_utils.*`
  (`TargetIsHexagon`) and `src/hexagon/op/builtin.*` (the HMX intrinsics)
- `src/hexagon/codegen/rt_mod_hexagon.cc` — registers `target.build.tilelang_hexagon`
- `src/hexagon/op/*.cc` — per-op lowering impls

A few shared passes still carry Hexagon hooks (vectorizer width, VTCM alignment,
attribute propagation, the host/device split); grep `kDLHexagon` / `"hexagon"` to
see them all.

---

## 5. The lowering flow (DSL → on-device)

```
@T.prim_func ─► TIR
   │  BackendContext.lower() → HexagonPassPipelineBody  (LowerOpaqueBlock, LowerTileOp→HMX/HVX, ...)
   │  SplitHostDevice  (device PrimFunc; ORDER params by host decl order — see §7)
   ▼
CodeGenTileLangHexagon  ─►  C + HVX/HMX intrinsics   (the generated kernel)
   +  src/tl_templates/hexagon/*.h                   (the hardware recipe)
   ▼
fastrpc.write_project   ─►  FastRPC skel (.idl/.c/.cc) + host + CMake
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
| `T.Kernel(..., num_workers=N)` | one `tl_parallel_for` over the linearized grid: up to N HW threads run its tile programs, each worker with a private VTCM arena | `worker.h` |
| `T.alloc_shared` | a byte offset into the tile program's VTCM arena (`tl_vtcm_arena()+off`, or the worker's `tl_arena+off`) | `vtcm.h` |
| `T.copy(matrix, crouton_shared)` | one `tl_crouton_copy`: pack a region of a strided fp16/fp32 matrix (DDR or VTCM) into Crouton tiles | `copy.cc`, `crouton.h` |
| `T.copy(crouton_shared, matrix)` | one `tl_crouton_copy`: unpack a Crouton region into a strided matrix | `copy.cc`, `crouton.h` |
| `T.gemm(A,B,C)` | explicit HMX atom loops; each result stores directly into its 2 KB-aligned native C tile | `gemm_hmx.py`, `hmx.h` |

Reading generated C for a tiled matmul now shows the grid loops, one
`tl_crouton_copy` per matrix at the DDR/Crouton boundary, Crouton VTCM offsets,
and explicit HMX atoms.

---

## 7. The hard-won lessons (gotchas worth knowing before you touch the recipe)

- **Crouton layout.** HMX reads operands from VTCM in a mandatory 32×32-tile layout:
  `cpos(i,j) = (i&~1)*32 + j*2 + (i&1)` — a **row-pair interleave**. That interleave
  is exactly one HVX `Q6_W_vshuff_VVR` of two source rows per 64-column span (and
  `vdeal` to unpack). `GemmHMX.infer_layout` assigns a `CroutonLayout` to every
  operand. A `T.copy` between any region of a Crouton buffer and a strided fp16 or
  fp32 matrix lowers to one `tl_crouton_copy`, which converts on the way. Row pairs
  of a row-contiguous matrix move as whole registers (masked at the region's
  edges); packing a column-contiguous fp16 matrix (a transposed operand in
  row-major storage) goes through a 32x32 word transpose in registers; other
  strides go element by element. DMA remains a
  separate row-major staging operation: the DMA descriptor can move
  contiguous/row-strided rectangles but cannot express the Crouton permutation.

- **DMA primitives are invoked explicitly.** `dma.h` provides raw User-DMA
  instruction wrappers, 1D/type-9 2D descriptor construction, cache maintenance,
  and a caller-owned descriptor ring with start/link/poll/wait/pop/flush operations.
  The queue allocates nothing and does not acquire or reset DMA0; its caller must
  own and serialize the engine. Synchronous 1D/2D copy helpers remain available
  through explicit extern calls. `T.dma_copy` lowers to IR builtins: the
  descriptors of one transfer (`hexagon_dma_submit`; a transfer over the
  descriptor limits takes several) and one `hexagon_dma_commit`, so every copy is
  one group of the tile program's queue. `T.dma_wait(n)` (`hexagon_dma_wait`)
  completes the oldest groups until at most n are pending. `PlanDMAQueue` bounds
  the descriptors and groups each tile program keeps in flight, declares the
  program's queue with those capacities (`hexagon_dma_queue`), and drains it at
  the end. The bound is conservative across branches; a loop that never waits is
  rejected, as are more than 256 transfers in flight. Regions must have equal scalar dtypes, static
  positive extents/strides, unit innermost stride, and provable bounds. Unsupported
  async copies fail lowering. Managed async kernels disable automatic VTCM
  allocation reuse and require a manual serial schedule and exclusive DMA0
  ownership; compiler pipelines and mixing raw DMA calls are rejected. Scope-based
  cleanup drains DMA and releases an acquired HMX atom lock on errors; these DSP
  projects compile with `-fno-exceptions` to avoid unsupported unwinder symbols.
  `T.copy` has no DMA-specific lowering; a
  Crouton endpoint makes it a `tl_crouton_copy`. In a `num_workers` kernel every tile
  program owns its DMA queue; raw DMA calls are rejected there.

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

- **VTCM is one arena per tile program.** `MergeSharedMemoryAllocations` lays out a
  tile program's shared buffers in one allocation and reuses the space of buffers
  whose lifetimes do not overlap; a DMA buffer stays live until the wait that
  completes it. The codegen binds the arena to the VTCM base and publishes
  `tl_vtcm_shared_high_water` for embedding/export capacity checks. Under the worker
  pool, `tl_parallel_for` gives every worker its own slice and runs no more workers
  than slices fit in the grant; `VerifyVTCMArena` rejects at compile time a
  `num_workers` whose slices exceed the target's `vtcm-capacity`.
  HMX atoms and Crouton copies use explicit shared buffers; there is no hidden
  top-down GEMM scratch allocation.

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

- **The grid is a loop, so allocations escape it.** `StorageRewrite` hoists buffers
  out of the grid loops to function scope. `MaterializeGrid` sinks every buffer that
  only the tile program uses back into it; without that, the workers of a
  `num_workers` kernel would share one copy.

---

## 8. Extending the backend (a recipe for the next developer)

- **Add a tile-op** (e.g. a reduction): add `src/hexagon/op/<op>.cc` with a C++ impl
  matching `TargetIsHexagon` (reuse the CPU lowering if it auto-vectorizes onto HVX);
  add it to the CMake GLOB (touch `CMakeLists.txt` to force re-glob).
- **Add a hardware recipe** (e.g. a new HMX kernel): put the C in
  `tl_templates/hexagon/*.h`, call it via `T.call_extern(...)` from an op's `lower`,
  and make `fastrpc.gen_dsp` `#include` the header when the source mentions its name
  (the lazy-include scheme). No compiler rebuild needed.
- **Make it idiomatic** (a tile-op instead of `call_extern`): register a
  `GemmBase`-style impl + `register_gemm_impl` (see `gemm_hmx.py`), so `T.gemm` lowers
  to it directly.
- **Surface a device-side failure** (VTCM/HMX unavailable, an unmet precondition):
  the generated kernel entry returns an `int32` status — `TL_OK` / `TL_ERR_VTCM` /
  `TL_ERR_HMX` / `TL_ERR_DMA` (`tl_templates/hexagon/common.h`). In a `num_workers` kernel each tile
  program returns its code, and `tl_parallel_for` returns the lowest failing worker's. The
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
automatic software-pipeline DMA lowering; and schedules for blocks larger than VTCM.
Worker-local managed DMA and synchronous parallel layout copies are now supported
within the [validated implementation boundaries](hexagon_dsl_demo_ops_zh.md#实现边界).
Explicit DMA prefetch alone does not establish cross-stage HMX/HVX overlap.

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

### HMX K-tile packets

`HMXIntrinEmitter.mma_tile()` reduces K for one output tile in DSL chunks of
at most 32 contiguous native tiles. Each `T.hmx.mma` intrinsic carries
an explicit tile count and matching operand read extents; `tl_hmx_mac_tiles`
emits one `:deep` packet and requires `1 <= n <= 32`. The caller handles the
final partial chunk. `mma_atom()` and `T.hmx.mma()` default to one tile
for existing explicit schedules. Accumulator acquisition, clear, conversion,
and output storage remain separate operations.
