# Offline matmul — generated kernel → SDK build → run on NPU → compare golden

The offline counterpart to [`../example_matmul.py`](../example_matmul.py). That one
calls `tilelang.compile` and does build + deploy + run in one shot. Here the **tilelang
codegen is done in advance and committed** (`project/`), so the build + run + compare
path needs **only the Hexagon SDK + adb + numpy — not tilelang**.

```
offline_matmul/
  project/        # the generated FastRPC project (IDL + skel + host + agent + CMake) — COMMITTED
  generate.py     # run-in-advance (needs tilelang): regenerate project/ when you change the kernel
  golden.py       # (numpy) write project/A.bin, project/B.bin, project/ref.npy
  reproduce.sh    # golden -> build_cmake (DSP + host) -> adb push/run -> verify   (no tilelang)
  verify.py       # compare project/C.bin against project/ref.npy
```

## Prerequisites (to build + run)

- The **Hexagon SDK build env** on `PATH` (`build_cmake`, `qaic`, `hexagon-clang++`, the
  Android NDK) — source your SDK's `setup_sdk_env.source` (on this host: `source /tmp/hexenv.sh`).
- **python + numpy** — for the golden inputs and the comparison (no tilelang).
- An **authorized adb device** (`adb devices`), e.g. a Snapdragon 8 Elite / Hexagon v79.

> Only `generate.py` needs **tilelang** — and it's already been run; its output is the
> committed `project/`. Re-run it (`python generate.py`) only if you change the kernel.

## Run

```bash
bash reproduce.sh                      # arch defaults to v79
# HEXAGON_DSP_ARCH=v73 bash reproduce.sh   # override the DSP arch
```

Expected output:

```
== 1/4  generate project + golden (tilelang lower -> FastRPC project) ==
== 2/4  build DSP skel + aarch64 host with the Hexagon SDK ==
   skel = project/hexagon_ReleaseG_toolv19_v79/libtl_matmul_kernel_skel.so
   host = project/android_ReleaseG_aarch64/tl_matmul_kernel_test
== 3/4  push + run on the NPU (one-shot FastRPC host driver) ==
== 4/4  compare device output vs golden reference ==
offline matmul 256x256x256 on HMX vs golden: max abs err = 0.0009761  (PASS)
```

## What each step does

0. **generate (pre-done, committed)** — `python generate.py` ran `tilelang.lower(..., target="hexagon")`
   (Layers 1–3 → cDSP C) and `fastrpc.write_project` (Layer 5 → FastRPC project), then patched the
   `CMakeLists` include to a repo-relative path. Its output is the committed `project/` +
   `generated_matmul_kernel.c`. The checked-in artifacts use one logical `T.copy` per
   matrix boundary between DDR and Crouton tiles, and the explicit-atom `T.gemm`
   lowering over Crouton buffers:

   ```c
   int32_t matmul_kernel(half* A, half* B, half* C) {
     for (bx = 0; bx < 4; ++bx) {
       for (by = 0; by < 4; ++by) {
         uint8_t* buf_dyn_shmem = (uint8_t*)tl_vtcm_arena();      // VTCM arena
         void* A_hmx = tl_vtcm_exclusive(buf_dyn_shmem + 0); ...  // C_hmx at +67584
         /* one logical T.copy: DDR matrix region -> Crouton tiles in VTCM */
         tl_crouton_copy<true, false>(A_hmx, 2, 8, 0, 0, 64, 256, A + by*16384, 256, 1);
         tl_crouton_copy<true, true>(B_hmx, 8, 2, 0, 0, 256, 64, B + bx*64, 256, 1);
         tl_hexagon_hmx_acc_acquire(acc);
         /* for each (mt,nt): clear; for kt: mma_atom; convert;
            store(C_hmx[mt,nt]); // every output tile is 2 KB aligned */
         tl_hexagon_hmx_acc_release(acc);
         /* one logical T.copy: Crouton tiles in VTCM -> DDR matrix region */
         tl_crouton_copy<false, false>(C_hmx, 2, 2, 0, 0, 64, 64,
                                       C + by*16384 + bx*64, 256, 1);
       }
     }
     return TL_OK;                                                 // int32 status ABI
   }
   ```

1. **golden** — `golden.py` (numpy) writes fixed-seed fp16 `A.bin`/`B.bin` and the fp32
   reference `ref.npy` into `project/`.
2. **build** — `build_cmake hexagon DSP_ARCH=v79` runs `qaic` on the IDL and compiles the
   skel with `hexagon-clang++ -mhmx -mhvx` → `libtl_matmul_kernel_skel.so` (cDSP ELF);
   `build_cmake android` builds the aarch64 host driver `tl_matmul_kernel_test`.
3. **run** — push the skel + host + `A.bin`/`B.bin`, then run the one-shot host driver. It
   opens the FastRPC session (the skel's `_open` acquires HMX/VTCM once), reads A/B, calls
   `matmul_kernel` on the cDSP, and writes `C.bin` (only on a `TL_OK` return — a nonzero
   `int32` status becomes `AEE_EFAILED` and the driver exits 1).
4. **verify** — `C.bin` vs the fp32 reference `A @ B`; `~1e-3` is fp16 rounding.

## Notes

- The host driver takes the buffers as argv in **param order**: `A.bin B.bin C.bin`
  (A in, B in, C out path).
- This uses the **one-shot** host (`_test`); the project also builds `tl_matmul_kernel_agent`,
  the resident socket agent that the Python adapter uses for its `~5 ms/call` path.
- If `hexagon-clang` errors on `-march=nocona` / a host `-isystem`, a host toolchain env
  (e.g. conda compilers) is leaking `CFLAGS` into the cross-compile — `reproduce.sh` already
  unsets them before building.
- VTCM is held per session, so `reproduce.sh` `pkill`s any resident agent first; if a run
  reports an open/VTCM error, clear stale agents: `adb shell pkill -f _agent`.
