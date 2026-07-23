# 从 TileLang 算子到 llama.cpp：一个单 Shape Walkthrough

这个目录只保留一个 Q4/HMX shape：`M=32, N=256, K=2048`。目标不是覆盖完整模型，而是让每一层
接线都能一眼对应。其他 shape 不匹配时，llama.cpp 自动继续 stock kernel。

## 0. 五个文件对应五个边界

```text
step1_tilelang_kernel.py
  TileLang PrimFunc: pack / dequant / explicit HMX compute
               |
               v  tilelang.lower(target="hexagon")
step2_generate.py
  generated/kernel_walkthrough_q4_m32_n256_k2048.cc + manifest
               |
               v  copy into ggml/src/ggml-hexagon/htp
step3_tl_ggml_adapter.cc
  matches() + borrowed VTCM/DMA/workers + calls generated entries
               |
               v
step4_llama_cpp.patch
  CMake source list + hmx_mm_2d_f32 tl_op_ctx/tl_dispatch seam
               |
               v  Hexagon clang/link
libggml-htp-v79.so
```

其中只有前两步由 TileLang 自动完成。step 3 是 framework adapter，step 4 是 llama.cpp 一次性接入点。

## 1. 先直接看完整 trace

```bash
cd /home/xwh/tilelang-hexagon/hexagon_backend_demo
./scripts/demo.sh llama-walkthrough
```

它会实际 lower kernel，然后按顺序显示：

1. 三个 TileLang PrimFunc；
2. generated C 的三个 entry 和底层 Q4/HMX atom；
3. manifest 中的 shape、symbol 和资源 owner；
4. adapter 的 `matches/run/register`；
5. patch 中的 CMake source list 和 stock dispatch seam。

`verify` 还会在当前机器已有 v79 `compile_commands.json` 时，用真实 Hexagon clang 参数编译 generated
kernel 和 adapter；没有配置过 llama.cpp build 时只跳过这一项。

也可以直接在本目录执行：

```bash
./run.sh generate
./run.sh trace
./run.sh verify
./run.sh install-commands
```

## 2. Step 1：TileLang 写的是什么

[`step1_tilelang_kernel.py`](step1_tilelang_kernel.py) 定义三个真实 PrimFunc：

| PrimFunc | 输入/输出 | TileLang 保留的调度 |
|---|---|---|
| `make_pack_stage` | llama FP32 activation -> A Crouton | K tile、row pair、`T.Layout` |
| `make_dequant_stage` | padded Q4 tiles -> B Crouton | runtime tile range、Q4 tile traversal |
| `make_compute_stage` | A/B Crouton -> FP32 output | HMX acquire/clear/mma/convert/store |

其中 `Q4HMXIntrinEmitter.dequant_tile` 和 `HMXIntrinEmitter.mma_atom` 是最底层硬件 atom；TileLang
仍然看得见 loop、layout、阶段和依赖关系。

## 3. Step 2：生成什么

```bash
./run.sh generate
```

[`step2_generate.py`](step2_generate.py) 对三个 PrimFunc 调用：

```python
tilelang.lower(
    func.with_attr("global_symbol", symbol),
    target=determine_target("hexagon"),
    enable_host_codegen=False,
    enable_device_compile=False,
)
```

输出：

```text
generated/
├── kernel_walkthrough_q4_m32_n256_k2048.cc
└── kernel_walkthrough_q4_m32_n256_k2048.json
```

generated C 提供三个普通 C ABI symbol：

```text
tl_walkthrough_q4_m32_n256_k2048_pack_kernel
tl_walkthrough_q4_m32_n256_k2048_dequant_kernel
tl_walkthrough_q4_m32_n256_k2048_compute_kernel
```

它不知道 llama.cpp，也不申请 FastRPC、VTCM、DMA 或线程。

## 4. Step 3：adapter 做什么

[`step3_tl_ggml_adapter.cc`](step3_tl_ggml_adapter.cc) 是可编译的单 shape adapter：

1. `extern "C"` 声明 generated symbols；
2. `matches()` 只接受 Q4_0、`32xNx2048`、`N % 256 == 0`、无 bias；
3. 在 llama.cpp 已有 VTCM 中规划 A/B/C 和双 weight stage；
4. 借已有 2D DMA 把每行 576B Q4 tile pad 到 640B；
5. 借已有 worker pool 并行调用 generated `dequant(begin,end)`；
6. lock 已 acquire 的 HMX resource，调用 generated `compute`；
7. constructor 把 `matches/run` 注册到 `tl_dispatch`。

这里没有第二个 session、第二个 VTCM grant 或第二个 worker pool。

## 5. Step 4：llama.cpp 改哪两处

[`step4_llama_cpp.patch`](step4_llama_cpp.patch) 只改两类位置。

### 5.1 编译期：HTP CMake

把下面两个 TU 加进原 DSP skel：

```cmake
tl_walkthrough_adapter.cc
kernel_walkthrough_q4_m32_n256_k2048.cc
```

并加入 TileLang Hexagon header path。generated TU 使用 `-fno-lto`，避免大型 HVX 表达式再次进入
whole-skel LTO。

### 5.2 运行期：stock matmul 前增加 offer

位置是 `matmul-ops.c::hmx_mm_2d_f32`：

```c
struct tl_op_ctx octx = {0};
octx.vtcm_base = ctx->vtcm_base;
octx.dma       = borrowed_dma_callbacks;
octx.parallel  = borrowed_worker_pool;
octx.dst = dst;
octx.act = activation;
octx.weight = weight;
octx.m = m; octx.n = n; octx.k = k;

if (tl_dispatch(&octx) == 0) return 0;
// 原 stock hmx_mm_2d_f32 从这里继续，代码不删除。
```

所以 TileLang 不匹配、资源不足或 stage 失败时，完整 stock 实现仍会执行。

## 6. Step 5：复制到 clean llama.cpp

```bash
TL=/path/to/tilelang-hexagon
DEMO="$TL/hexagon_backend_demo/walkthrough/llama_cpp_q4"
LCPP=/path/to/clean/llama.cpp
HTP="$LCPP/ggml/src/ggml-hexagon/htp"

cd "$DEMO"
./run.sh generate

cp generated/kernel_walkthrough_q4_m32_n256_k2048.cc "$HTP/"
cp step3_tl_ggml_adapter.cc "$HTP/tl_walkthrough_adapter.cc"
git -C "$LCPP" apply --check "$DEMO/step4_llama_cpp.patch"
git -C "$LCPP" apply "$DEMO/step4_llama_cpp.patch"
```

`git apply --check` 必须先通过；不要对已经集成过 TileLang 的 llama.cpp tree 再应用此最小 patch。

## 7. Step 6：构建 DSP skel

```bash
export TILELANG_SOURCE_DIR="$TL"
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.6.0.0
export HEXAGON_TOOLS_ROOT=/path/to/HEXAGON_Tools/19.0.07
export ANDROID_NDK_ROOT=/path/to/android-ndk-r25c

cmake --preset arm64-android-snapdragon-release \
  -S "$LCPP" -B "$LCPP/build-walkthrough" \
  -DGGML_HEXAGON_TILELANG_WALKTHROUGH=ON \
  -DGGML_HEXAGON_TILELANG_SOURCE_DIR="$TL"

cmake --build "$LCPP/build-walkthrough" --target htp-v79 -j8
```

最终生成的仍是 llama.cpp 原来的 `libggml-htp-v79.so`。这里没有额外运行时动态库。

## 8. Step 7：部署与验证

```bash
DEVICE=/data/local/tmp/llamahtp
SKEL="$LCPP/build-walkthrough/ggml/src/ggml-hexagon/libggml-htp-v79.so"

adb shell "test -f $DEVICE/libggml-htp-v79.so.stock || \
  cp $DEVICE/libggml-htp-v79.so $DEVICE/libggml-htp-v79.so.stock"
adb push "$SKEL" "$DEVICE/libggml-htp-v79.so"

adb shell "cd $DEVICE && LD_LIBRARY_PATH=. \
  ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp \
  ./llama-bench -m LFM2-1.2B-Q4_0.gguf -dev HTP0 -ngl 99 -p 32 -n 0 -r 1"

adb shell "cp $DEVICE/libggml-htp-v79.so.stock $DEVICE/libggml-htp-v79.so"
```

这个最小 build 只会接管 K=2048 的单 shape 路径；模型中的其他 matmul 继续 stock。项目中的完整 Q4
集成则额外生成 K=8192、N=128/512 family 和 staged fast path。

## 9. 从这个例子推广到通用 backend

固定 shape 只是为了说明接线。通用方案应把下面两项继续自动化：

1. 根据 manifest 自动生成 symbol table、`matches()` skeleton 和 CMake source list；
2. 一个 DSP skel 只保留一个 shared registry，使 Q4、Q8 和其他 generated op 可以同时注册。

不应自动化的是宿主资源策略本身：llama.cpp 的 DMA queue、worker pool、VTCM lifetime 和 fallback
仍由 llama.cpp 持有，TileLang 只通过稳定的 resource-lease ABI 使用它们。
