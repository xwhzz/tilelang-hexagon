# Offline matmul — generated kernel → SDK build → run on NPU → compare golden

The offline counterpart to [`../example_matmul.py`](../example_matmul.py). That one
calls `tilelang.compile` and does build + deploy + run in one shot; here we **stop at
codegen** and drive the bare Hexagon SDK by hand, so you can see the generated C as a
buildable FastRPC project and run it on the device without tilelang in the loop.

```
offline_matmul/
  generate.py     # tilelang lower -> ./project/ (IDL+skel+host+agent+CMake) + golden A.bin/B.bin/ref.npy
  reproduce.sh    # generate -> build_cmake (DSP+host) -> adb push/run -> verify
  verify.py       # compare project/C.bin against project/ref.npy
  project/         # generated FastRPC project + build output  (git-ignored; reproduce with generate.py)
```

## Prerequisites

- The **Hexagon SDK build env** on `PATH` (`build_cmake`, `qaic`, `hexagon-clang++`, the
  Android NDK) — source your SDK's `setup_sdk_env.source` (on this host: `source /tmp/hexenv.sh`).
- **tilelang** importable + **numpy** — only for step 1 (the codegen). On this host that's
  the `tl` conda env.
- An **authorized adb device** (`adb devices`), e.g. a Snapdragon 8 Elite / Hexagon v79.

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

1. **generate** — `tilelang.lower(..., target="hexagon")` (Layers 1–3) produces the cDSP C;
   `_fastrpc.write_project` (Layer 5) wraps it into a FastRPC project. The generated device
   kernel is also written to `generated_matmul_kernel.c` for inspection — the whole `T.gemm`
   is one call:

   ```c
   int32_t matmul_kernel(half* A, half* B, half* C) {
     for (bx = 0; bx < 4; ++bx) {
       uint8_t* buf = (uint8_t*)((char*)tl_vtcm_base() + 2048);   // VTCM arena
       void* A_sh=buf+0; void* B_sh=buf+32768; void* C_sh=buf+65536;
       for (by = 0; by < 4; ++by) {
         /* T.copy A,B -> VTCM as HVX half8 loops */
         tl_hexagon_hmx_gemm(C_sh, A_sh, B_sh, 64, 64, 256, 0, 0);  // -> HMX matrix engine
         /* T.copy C_sh -> global */
       }
     }
     return TL_OK;                                                 // int32 status ABI
   }
   ```

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
