# Integrating a tilelang kernel into llama.cpp (ggml-hexagon)

How a tilelang-authored Hexagon kernel is made to run **inside** llama.cpp's
`ggml-hexagon` NPU backend, on real model weights, as a switchable replacement
for a stock backend kernel. Written from the M2b work: a tilelang q4_0 matmul
wired into LFM2-1.2B's forward pass, device-validated for correctness.

> Status: this is a **correctness proof-of-concept**, not a productized path.
> The [Reflection](#reflection-is-this-general-is-it-clean) section is deliberately
> critical about what is reusable vs. what is scaffolding.

---

## 1. Two ways to run a tilelang Hexagon kernel

There are two fundamentally different execution modes, and the whole shape of the
integration follows from picking the second one.

**Mode A — standalone FastRPC skel** (what `tilelang.compile(target="hexagon")`
and `examples/hexagon/*` produce). `tilelang.lower` → cDSP C → a *complete FastRPC
project*: its own `_skel.so`, its own `_open` handler that **acquires VTCM + HMX +
power for the session**, its own entry point. Great for authoring, unit-testing,
and benchmarking a kernel in isolation. But it is a *separate loadable module with
its own resource session* — you cannot drop it inside another process's skel.

**Mode B — embedded in a host skel** (what llama.cpp needs). llama.cpp already has
its own DSP skel (`libggml-htp-v79.so`) that owns the FastRPC session, the VTCM
region, the HMX lock, and a worker pool. To have a tilelang kernel run *as one op
of that graph*, the kernel must be **co-compiled into the host skel as source, and
must share the host's already-acquired resources** — it must not open its own
session or re-acquire VTCM/HMX.

Mode B is the integration. Everything below is about making a tilelang kernel a
well-behaved guest inside someone else's skel.

---

## 2. Anatomy of the embedded integration

Reusable pieces live in the **tilelang repo** (`src/tl_templates/hexagon/`);
op-specific glue lives in the **llama.cpp tree** (`ggml/src/ggml-hexagon/`):

| file | role | provenance |
|---|---|---|
| `tl_templates/hexagon/tl_embed.h` | reusable, C-safe **embedding API**: `tl_op_ctx` + `tl_op_desc` + registry decls | **tilelang repo** |
| `tl_templates/hexagon/tl_bridge.h` | reusable **bridge**: bind the runtime to the host VTCM + ride its HMX lock | **tilelang repo** |
| `tl_templates/hexagon/hmx.h` | the **fp16 HMX gemm runtime** (Crouton pack → `mxmem` MAC → unpack) | **tilelang repo** (reused verbatim) |
| `htp/tl_ggml_matmul.cc` | the registry impl + the q4_0 matmul op (`matches`/`run`, self-registers) | integration, hand C++ |
| `htp/matmul-ops.c` | generic intercept: build a `tl_op_ctx`, call `tl_dispatch()` (~6 lines) | integration |
| `htp/CMakeLists.txt` | add the `.cc`, `-I` the tilelang templates, `-Wno-unused-function` | integration |

### Data flow

```
llama.cpp graph (host, ARM64)
 └─ ggml-hexagon backend: a MUL_MAT node, q4_0 weight
      └─ FastRPC ─► ggml-htp skel (cDSP)
           op_matmul ─► hmx_mm_2d_f32(ctx, dst, act, weight, m,k,n, …)
                │   ┌──────────── intercept (8 lines) ─────────────┐
                └──►│ if (tl_mm_enabled && q4_0 && no fused bias)   │
                    │   tl_ggml_matmul_q4_0(ctx->vtcm_base,         │
                    │                       ctx->vtcm_size,         │
                    │                       ctx->vtcm_rctx, …)      │
                    │     ├ HAP_compute_res_hmx_lock(rctx) ← take   │
                    │     ├ tl_bridge_enter(vtcm_base, size)        │
                    │     ├ for each N-chunk (to fit VTCM):         │
                    │     │    tl_dequant_q4_0_chunk() ← hand C     │
                    │     │    tl_hexagon_hmx_gemm()   ← hmx.h      │
                    │     │    write f32 output                     │
                    │     ├ tl_bridge_exit()                        │
                    │     └ HAP_compute_res_hmx_unlock(rctx)        │
                    │   return 0  (handled)                         │
                    │ else  return -1  → stock kernel runs below    │
                    └───────────────────────────────────────────────┘
```

---

## 3. The runtime bridge (resource sharing)

`hmx.h`/`vtcm.h` were written for Mode A: they lazily **acquire** the VTCM region
(`tl_vtcm_acquire` via `HAP_compute_res`) and set up an HMX session
(`tl_hmx_session_init`: power + VTCM + HMX + per-thread enable + fill the fp16
unit-scale tile at `base+0`). Inside the host skel, all of that is **already
owned by the host** — re-acquiring is at best wasteful and at worst a fault.

The bridge makes the tilelang runtime *ride* the host's resources instead:

```c
// tl_ggml_bridge.h
static inline void tl_bridge_enter(void *vtcm_base, unsigned int vtcm_size) {
  tl_vtcm_base_ptr = (uint8_t *)vtcm_base;   // → tl_vtcm_acquire() now no-ops
  tl_vtcm_total    = vtcm_size;
  tl_hmx_fill_unit_scales((uint32_t *)tl_vtcm_base_ptr);  // our scales @ base+0
  tl_hmx_inited    = 1;                        // trust the caller's HMX lock
}
```

Two facts make this safe and were the load-bearing discoveries:

- **VTCM**: `tl_vtcm_acquire()` is idempotent on a set base pointer — binding
  `tl_vtcm_base_ptr` to `ctx->vtcm_base` makes every later `tl_vtcm_base()` return
  the host's region with no acquire. A `MM_SELECT`-style op *replaces* the stock
  kernel, so nothing else touches VTCM for the op's duration.
- **HMX**: the host locks HMX with `HAP_compute_res_hmx_lock(ctx->vtcm_rctx)`.
  `hmx_mm_2d_f32` takes that lock *internally* (not at its top), so the intercept
  runs with HMX **unlocked** — the adapter therefore takes the lock itself with the
  host's `rctx` (one combined VTCM+HMX resource, `main.c:vtcm_alloc`). No second
  `HAP_compute_res_acquire` (which is the thing that would have failed).

Gotchas banked while proving this on-device:
- VTCM is only usable **after** `HAP_compute_res_acquire_cached` sets
  `ctx->vtcm_valid` (in `vtcm_acquire`), *not* right after `vtcm_alloc` — touching
  it earlier SIGABRTs.
- `hmx.h`'s state is `static`, so the bridge and the generated/authored kernel that
  reads it **must live in the same translation unit**.
- DSP `FARF` needs a `<exe>.farf` config to reach `logcat`; without it, use an
  `abort()`/return-code signal for device correctness proofs.

---

## 4. Dispatch: a data-driven registry (not a hardcoded `if`)

The host's stock kernel doesn't name any tilelang op. It builds a `tl_op_ctx` and
asks the registry:

```c
// matmul-ops.c, top of hmx_mm_2d_f32(...)
struct tl_op_ctx octx = { ctx->vtcm_base, ctx->vtcm_size, ctx->vtcm_rctx,
                          TL_OP_MATMUL, weight_type, (src2 != NULL),
                          dst, activation, weight, m, k, n, act_stride, dst_stride };
if (tl_dispatch(&octx) == 0) return 0;   // a tilelang op handled it; else fall through
```

A tilelang op is a `tl_op_desc { name, matches, run }` that **self-registers** at
skel load via `__attribute__((constructor))`; `tl_dispatch` walks the table,
running the first op whose `matches()` accepts the ctx. `run()` returns `-1` to
decline (e.g. `m > 32`, fused bias, odd shapes) and the stock kernel runs.
**Adding a kernel is a new self-contained `.cc` with its own descriptor — no edit
to any stock function.** `tl_mm_enabled` is the master A/B toggle (verified
on-device: the constructor fires and dispatch routes the op inside the skel).

---

## 5. Build & A/B

`htp/CMakeLists.txt`:
```cmake
include_directories(/path/to/tilelang-hexagon/src
                    /path/to/tilelang-hexagon/src/tl_templates/hexagon)
add_library(${HTP_LIB} SHARED  … matmul-ops.c  tl_ggml_matmul.cc)
set_source_files_properties(tl_ggml_matmul.cc PROPERTIES COMPILE_OPTIONS "-Wno-unused-function")
```
Rebuild the `htp-v<arch>` external project, `adb push` the skel, run `llama-cli`.
A/B by flipping `tl_mm_enabled` (0=stock, 1=tilelang) and rebuilding. Verified on
LFM2-1.2B: prefill 156→0.8 t/s with **both coherent** — the kernel is provably
active *and* correct.

---

## 6. What is actually "tilelang" here

Be precise about it: **the reused tilelang asset is `hmx.h` — the fp16 HMX gemm
runtime** (`tl_hexagon_hmx_gemm`: HVX-pack row-major → Crouton, the locked
`mxclracc`/`mxmem` MAC, HVX-unpack). That is the hard, valuable, DSL-derived part,
and it dropped into the host toolchain cleanly.

Everything *around* it in `tl_ggml_matmul.cc` — the q4_0 dequant, the N-chunk
tiling, the M-padding, the f32↔fp16 conversions, the output write — is **hand-
written C++**. In the standalone Mode-A validation (`qmatmul_validate.py`) all of
that *was* tilelang-generated (dequant loops + `T.gemm`), but that artifact is a
whole FastRPC skel; it can't be embedded. So for Mode B the surrounding op was
re-authored by hand and only the runtime template was reused.

---

## Reflection: is this general? is it clean?

Short answers: **the bridge is general and clean; the op wiring is a proof-of-
concept and is not.**

**What is genuinely reusable / clean**
- The *bridge pattern* (bind the tilelang runtime globals to a host-owned VTCM +
  ride the host HMX lock, no self-acquire) is small, correct, and works for **any**
  tilelang HMX/HVX kernel embedded in a host skel — not specific to matmul or q4_0.
- Reusing `hmx.h` verbatim in a foreign toolchain (it compiles clean under the
  backend's `hexagon-clang -mv79 -mhmx`) shows the runtime template is portable.
- The `return -1 → fall back to stock` contract is a clean, safe integration
  boundary: the tilelang path can be partial and never breaks the model.

**What is NOT general (the honest smells)**
1. **The DSL→drop-in story is only half-real.** What runs in llama.cpp is hand-C
   that *calls* the tilelang runtime, not a `tilelang.compile` artifact. A truly
   general path needs a codegen mode that emits an **embeddable kernel** (a plain
   `int fn(args, void* vtcm, …)` C function + the runtime headers), *not* a
   standalone FastRPC skel. That mode does not exist yet.
2. **Hardcoded intercept.** One `if` bolted to the top of `hmx_mm_2d_f32`. Each new
   op = another bespoke intercept in another stock function. There is no
   registration/dispatch seam (a table keyed by op+dtype+shape-predicate).
3. **`static` runtime state forces one TU.** Because `hmx.h` uses file-static
   globals (`tl_vtcm_base_ptr`, `tl_hmx_inited`, …), every embedded tilelang op that
   shares the runtime must be in the *same* `.cc`, or each TU gets its own copy of
   the state. That does not scale to many ops. A clean version threads an explicit
   `tl_ctx*` (VTCM base/size, HMX handle, scales ptr) through the runtime instead of
   globals.
4. **Hand-written dequant + tiling.** The op's data movement (repacked-tile
   dequant, N-chunk tiling, M-pad) is hand-C. Generality wants these expressed in
   tilelang and lowered — which needs (a) the embeddable-codegen from #1, and (b)
   shape-family support (the codegen emits fixed shapes; real ops span shapes, so
   today the adapter hand-loops).
5. **Absolute include paths** to the tilelang repo in `CMakeLists.txt`; not
   packaged (no installed headers, no find-package).
6. **Resource assumptions are implicit.** The bridge silently assumes it is called
   with HMX lockable via `rctx`, VTCM valid, and exclusive use of VTCM for the op.
   Those held for a `MM_SELECT` replacement but are undocumented preconditions, not
   enforced.

**A clean, general version would be:**
- a **`tilelang` "embeddable kernel" target** → emits `kernel.c` + a small manifest
  (name, dtype, shape predicate, VTCM footprint), no session/skel;
- the runtime (`hmx.h`/`vtcm.h`) refactored to take an **explicit context struct**,
  killing the static-globals/one-TU constraint;
- a **host-side registry** in the backend (`register_tl_op(predicate, fn)`) so
  dispatch is data, not a hardcoded `if`;
- a thin **`tl_ggml` shim library** (installed headers + the bridge) that the host
  backend depends on, instead of absolute `-I` paths;
- shape handling either via runtime-shape kernels or an adapter that the codegen
  emits, not hand-loops.

That is the roadmap from "we proved one kernel runs correctly inside llama.cpp" to
"a developer authors a kernel in tilelang and registers it."

### Update — the registry step (done, device-verified)

The first slice of the roadmap is implemented:
- **Op registry + explicit op-context** — `tl_embed.h` (`tl_op_ctx` + `tl_op_desc`
  + `tl_dispatch`) replaces the hardcoded `if`. Ops self-register at skel load; the
  host builds a ctx and dispatches. Smell **#2 (hardcoded intercept) is fixed**, and
  **#6 (implicit preconditions)** is now an explicit, documented struct.
- **Reusable shim in the tilelang repo** — `tl_embed.h` (C-safe API) + `tl_bridge.h`
  (the VTCM/HMX bridge) live under `src/tl_templates/hexagon/`, so the host depends
  on tilelang-provided headers, not on hand-copied files.

Still deferred (bigger, higher-risk):
- the **embeddable-kernel codegen target** (#1) — the op body is still hand-C that
  *calls* the tilelang HMX runtime, not a `tilelang.compile` artifact;
- the **`hmx.h`/`vtcm.h` core `tl_ctx` refactor** (#3) — threading a context struct
  through the runtime to kill the `static` globals ripples into the codegen and every
  example, so it was intentionally not done in this step; note the globals are *not*
  a correctness blocker as long as each op is a self-contained TU;
- **shape-family handling** (#4) and **installed/packaged includes** (#5, partial:
  the shim moved to the repo, but `CMakeLists.txt` still uses absolute `-I` paths).

## Perf reality

The proof kernel is slow by construction (scalar per-call dequant, M-pad). Profiling
showed the tilelang **HMX gemm itself is ~competitive (≈50 t/s, gemm-limited)** and
the **scalar dequant is ~99% of the cost**; making it fast needs hand-HVX dequant,
which reproduces the backend's tuned kernel → a **parity ceiling**, not a win. See
the perf note in `MEMORY`/the analysis: for LFM2 on this mature backend, decode is
memory-bound on weight streaming (backend-optimal), so the win-shaped targets are a
better quant format (fewer weight bytes) or an op/model the backend doesn't cover —
not re-implementing its q4_0 matmul.
