# TileLang 生成代码如何集成进 llama.cpp

本文给出当前 Q4/HMX 案例的完整源码级链路。Q4 只是例子；真正可复用的是
`generated device body + manifest + framework adapter + host resource lease` 这套嵌入模式。

第一次阅读建议先运行单 shape walkthrough：

```bash
cd /home/xwh/tilelang-hexagon/hexagon_backend_demo
./scripts/demo.sh llama-walkthrough
```

对应文件全部在 [`walkthrough/llama_cpp_q4/`](../walkthrough/llama_cpp_q4/README.md)，按
TileLang kernel、generator、adapter、llama.cpp patch 顺序编号。

## 1. 先说明当前自动化边界

当前流程不是 `tilelang.compile()` 直接修改 llama.cpp：

| 部分 | 当前由谁产生 | 是否通用 |
|---|---|---|
| Hexagon kernel body | TileLang lowering 自动生成 `.cc` | 是 |
| entry symbol、shape、VTCM、layout contract | emitter 自动生成 manifest | 是 |
| DSP embedding ABI | `tl_templates/hexagon/tl_embed.h` | 是 |
| ggml tensor/shape 到 embedding ABI 的转换 | `tl_ggml_matmul.cc` | framework-specific |
| ggml stock op 中的 dispatch seam | `ggml-hexagon.patch` | 一次性 framework patch |
| 把 generated TU 编进 DSP skel | CMake patch | 当前仍需显式列出 |

因此，**算子指令和 TileLang 调度是生成的，框架资源接线目前仍是手写的**。manifest 现在用于校验，
尚未自动生成 registry/CMake；这是后续 backend 产品化需要补齐的部分。

## 2. 最终进入 llama.cpp 的文件

```text
llama.cpp/ggml/src/ggml-hexagon/htp/
├── matmul-ops.c                         stock op + tl_dispatch seam
├── tl_ggml_matmul.cc                    framework adapter / registry
├── kernel_qmatmul_hmx_atoms_*.cc        TileLang 生成的完整 fallback family
└── kernel_qmatmul_hmx_staged_*.cc       TileLang 生成的 pack/dequant/compute stages

TileLang source tree/
└── src/tl_templates/hexagon/
    ├── tl_embed.h                       C-safe resource lease ABI
    ├── tl_bridge.h                      generated C 到 HMX/HVX recipe 的桥
    ├── qmatmul.h                        Q4 native dequant tile atom
    ├── hmx.h                            HMX instruction recipes
    └── vtcm.h                           VTCM runtime helpers
```

只有 DSP skel `libggml-htp-v79.so` 需要重编。ARM 侧 `llama-cli`、ggml graph 和 FastRPC IDL 都不变，
因为新 kernel 在已有 `hmx_mm_2d_f32` RPC 调用内部被分派。

## 3. 第一步：生成 embeddable Hexagon translation units

在集中交付目录中：

```bash
cd /home/xwh/tilelang-hexagon/hexagon_backend_demo
./scripts/demo.sh generate-q4
```

输出在 `results/generated/`。staged emitter 对每个 K 生成一个 `.cc` 和一个 manifest：

```text
kernel_qmatmul_hmx_staged_32x256x2048.cc
kernel_qmatmul_hmx_staged_32x256x2048.manifest.json
kernel_qmatmul_hmx_staged_32x256x8192.cc
kernel_qmatmul_hmx_staged_32x256x8192.manifest.json
```

一个 staged `.cc` 内有三个 `extern "C"` entry：

| entry | 生成 ABI | 职责 |
|---|---|---|
| `*_pack_kernel` | `(float *A, half *A_hmx, uint32_t *bias)` | FP32 activation -> A Crouton |
| `*_dequant_kernel` | `(uint8_t *W, half *B_hmx, int begin, int end)` | 处理 caller 分配的 Q4 tile range |
| `*_compute_kernel` | `(A_hmx, B_hmx, bias, C_hmx, float *C, int stride)` | 显式 HMX K-loop 和输出 unpack |

生成文件只包含普通 C/C++ 函数和 TileLang Hexagon headers，不包含 FastRPC session、线程池或设备部署
逻辑。参数顺序就是 ABI，不能按名字排序或在 adapter 中自行重排。

manifest 记录相同的 entry symbol 以及运行约束。例如 K=2048 的 staged path 声明：

```text
weight source:   [8][64][576] bytes
DMA destination: [8][64][640] bytes
A Crouton:       131072 bytes
B Crouton:       1048576 bytes
raw stages:      2 x 327680 bytes
workers:         caller-owned synchronous parallel_for
DMA:             caller-owned ordered 2D queue
HMX:             caller locks resource; generated compute owns accumulator protocol
```

## 4. 第二步：把生成文件和 adapter 放进 HTP skel 源码树

对 clean llama.cpp `4fc4ec5` 示例：

```bash
TL=/path/to/tilelang-hexagon
LCPP=/path/to/llama.cpp
OUT="$TL/hexagon_backend_demo/results/generated"
HTP="$LCPP/ggml/src/ggml-hexagon/htp"

cp "$OUT"/kernel_qmatmul_hmx_*.cc "$HTP/"
cp "$TL/examples/hexagon/llama_cpp_integration/tl_ggml_matmul.cc" "$HTP/"

git -C "$LCPP" apply --check \
  "$TL/examples/hexagon/llama_cpp_integration/ggml-hexagon.patch"
git -C "$LCPP" apply \
  "$TL/examples/hexagon/llama_cpp_integration/ggml-hexagon.patch"
```

patch 做三件事：

1. 外层 CMake 增加 `GGML_HEXAGON_TILELANG_Q4_HMX` 和 TileLang source path；
2. HTP CMake 把 adapter 与 generated `.cc` 编进 `libggml-htp-${DSP_VERSION}.so`；
3. 在 `hmx_mm_2d_f32` 的 stock 实现之前构造 `tl_op_ctx` 并调用 `tl_dispatch`。

generated translation units 使用 `-fno-lto`。这是必要的编译边界：大型 HVX nibble/dequant 表达式再次进入
whole-skel LTO 会显著增加 Hexagon LLVM 内存，而这些 TU 已经在 `-O2` 阶段完成目标代码生成。

## 5. 第三步：CMake 编译进 DSP skel

```bash
export TILELANG_SOURCE_DIR="$TL"
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.6.0.0
export HEXAGON_TOOLS_ROOT=/path/to/HEXAGON_Tools/19.0.07
export ANDROID_NDK_ROOT=/path/to/android-ndk-r25c

cmake --preset arm64-android-snapdragon-release \
  -S "$LCPP" -B "$LCPP/build-snap" \
  -DGGML_HEXAGON_TILELANG_Q4_HMX=ON \
  -DGGML_HEXAGON_TILELANG_SOURCE_DIR="$TL"

cmake --build "$LCPP/build-snap" --target htp-v79 -j8
```

已经配置好的本机 build 可以直接用：

```bash
cd /home/xwh/tilelang-hexagon/hexagon_backend_demo
./scripts/demo.sh build-q4
```

链接关系是静态 co-compile，不是运行时加载第二个 `.so`：

```text
generated *.cc
  -> include tl_bridge.h/qmatmul.h/hmx.h
  -> Hexagon clang -mhvx -mhmx
  -> linked with matmul-ops.c + tl_ggml_matmul.cc
  -> libggml-htp-v79.so
```

## 6. 第四步：stock op 如何把一次 matmul 交给 TileLang

集成点位于 `ggml/src/ggml-hexagon/htp/matmul-ops.c::hmx_mm_2d_f32`，在原 stock HMX
实现之前：

```c
struct tl_op_ctx octx = {0};
octx.vtcm_base = ctx->vtcm_base;
octx.vtcm_size = ctx->vtcm_size;
octx.vtcm_rctx = ctx->vtcm_rctx;
octx.dma = borrowed_dma_callbacks(ctx->dma[0]);
octx.parallel = borrowed_worker_pool(ctx->worker_pool, n_threads);
octx.kind = TL_OP_MATMUL;
octx.weight_type = weight_type;
octx.has_bias = (src2 != NULL);
octx.dst = dst;
octx.act = activation;
octx.weight = weight;
octx.m = m; octx.n = n; octx.k = k;
octx.act_stride = act_stride;
octx.dst_stride = dst_stride;

if (tl_dispatch(&octx) == 0) {
    return 0;             // TileLang handled the op
}
// Existing stock hmx_mm_2d_f32 continues unchanged.
```

`tl_op_ctx` 是借用资源，不转移所有权：

- VTCM 由 llama.cpp 已有 HTP session 申请；
- DMA queue 由 llama.cpp 创建和销毁，TileLang 只能 push/pop；
- worker pool 由 llama.cpp 持有，`parallel.run` 是同步 callback；
- HMX compute resource 已经 acquire，adapter 只在 compute stage 前后 lock/unlock；
- generated kernel 不允许再创建 FastRPC session、agent 或第二个 worker pool。

## 7. 第五步：adapter 如何选择并调用生成 symbol

`tl_ggml_matmul.cc` 在加载 skel 时通过 constructor 注册：

```c++
static const tl_op_desc desc = {
    "q4_0_hmx_atoms", tl_q4_hmx_matches, tl_q4_hmx_run};

__attribute__((constructor))
static void register_q4() { tl_register_op(&desc); }
```

`matches()` 当前只接受：

- `TL_OP_MATMUL`、`HTP_TYPE_Q4_0`、无 fused bias；
- `M > 0 && M % 32 == 0`；staged fast path 进一步要求 `M == 32`；
- `N % 128 == 0`，K 为生成表中的 2048 或 8192；
- activation contiguous in K，output stride 足够；
- VTCM、DMA 和 HMX resource 均有效。

`run()` 对 `M=32` 优先走 staged symbol table：

```text
pack(A) once
  -> async 2D DMA: each raw 576B Q4 tile -> aligned 640B VTCM row
  -> 6 borrowed workers call generated dequant(begin, end)
  -> lock single HMX
  -> generated compute(A_crouton, B_crouton, C)
  -> unlock HMX
  -> prefetch/consume next N=256 chunk
```

llama.cpp 传入的 Q4 weight 已经是 `[N/32][K/32][576]`，所以 adapter 不创建长期 weight repack
cache。TileLang dequant atom 直接把每个 native tile 展开到 HMX B Crouton。

## 8. fallback 为什么不会破坏 llama.cpp

返回值契约只有两个：

- `0`：某个 registered TileLang op 已完整处理，stock function 立即返回；
- `-1`：没有匹配或执行失败，控制流继续进入原 stock implementation。

不支持的 dtype、shape、bias、tail 和资源不足都必须 decline。staged 过程中只使用借来的 VTCM/DMA；
若中途失败，stock path 会重新计算并覆盖完整输出。`TL_Q4_HMX_ENABLED=0` 或 CMake option `OFF`
时不会改变 stock 行为。

## 9. 第六步：部署并验证真实模型

第一次部署前保留 stock skel，且不要覆盖已有备份：

```bash
adb shell "test -f /data/local/tmp/llamahtp/libggml-htp-v79.so.stock || \
  cp /data/local/tmp/llamahtp/libggml-htp-v79.so \
     /data/local/tmp/llamahtp/libggml-htp-v79.so.stock"
```

集中目录已经归档验证过的 skel，因此现场可直接运行：

```bash
cd /home/xwh/tilelang-hexagon/hexagon_backend_demo
./scripts/demo.sh deploy-q4
./scripts/demo.sh status
./scripts/demo.sh bench-q4
./scripts/demo.sh chat-q4
./scripts/demo.sh restore-stock
./scripts/demo.sh status
```

FastRPC loader 对当前目录敏感；模型命令必须在设备 `DEVICE_DIR` 内启动，并设置
`LD_LIBRARY_PATH=.` 与 `ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp`。`demo.sh` 已封装该规则。

## 10. 如何把这套模式扩展到其他算子或框架

对新算子，framework adapter 只应实现四件事：

1. 从宿主 tensor/op descriptor 构造稳定的 `tl_op_ctx`；
2. 用 `matches()` 定义 dtype、shape、layout 和资源前置条件；
3. 把宿主已有 DMA/worker/VTCM/HMX 以窄 callback ABI 借给 generated stages；
4. 任何未覆盖路径回落原实现。

下一步应把当前手写步骤收敛成通用工具：

```text
TileLang emitter
  -> kernel.cc + op.manifest.json
  -> manifest-driven registry generator
  -> tilelang_hexagon_embed_library(...) CMake helper
  -> one shared registry TU per DSP skel
```

这也解决当前 Q4/Q8 adapter 各自定义 registry、不能同时直接编进同一 skel 的限制。真正的 backend
完成标准不是继续增加模型专用 patch，而是让 manifest、registry、CMake 和资源 ABI 成为稳定接口。
