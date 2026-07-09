# tilelang Hexagon NPU 后端 —— 总览与现状

> 目标:**用 tilelang DSL 写高性能算子,把边缘 LLM(LFM2)跑到高通 Hexagon NPU 上** ——
> 通过把 tilelang 生成的 kernel 换进 llama.cpp 的 `ggml-hexagon` 后端。本文是顶层叙事,
> 细节文档见文末索引。英文版见 [`hexagon_backend_summary.md`](hexagon_backend_summary.md)。

---

## 1. 背景:Hexagon 不是 GPU

Hexagon 是 **VLIW DSP + HVX + HMX + VTCM + DMA**,和 GPGPU 之间有一条鸿沟:

| 部件 | 是什么 | 关键约束 |
|---|---|---|
| **HVX** | 1024-bit / 128-byte 向量单元,~6 个硬件线程 | **没有亚寄存器操作** —— 每条指令都是整个 128B 寄存器 |
| **HMX** | 一个 fp16 矩阵引擎,32×32 Crouton tile | **没有 mma 指令**;load 即 MAC;累加器不可见、全核唯一 |
| **VTCM** | 8MB 软件管理的 SRAM | 手动搬运(没有自动 cache) |
| **DMA** | 异步搬运 | —— |

没有 SIMT、没有自动 cache、没有 mma。下面几乎所有工作都是这条鸿沟的直接后果。

---

## 2. 整体栈(自底向上)

| 层 | 内容 | 位置 | 状态 |
|---|---|---|---|
| **Codegen** | 满寄存器 HVX 向量化(1024-bit 宽度 + 最窄 dtype 填满寄存器 + 原生 `ext_vector` 的 `vec_type`) | `src/transform/loop_vectorize.cc`、`src/tl_templates/hexagon/common.h`、`codegen_hexagon.cc` | ✅ 已提交,零回归 |
| **DSL kernel** | 全 DSL q4_0 反量化(追平手写 HVX)+ 反量化→VTCM→`T.gemm`→HMX 融合 matmul;紧凑常驻 scale | `examples/hexagon/example_qmatmul*.py` | ✅ 设备验证(rel 2.5e-4) |
| **HMX 抽象** | `HMXIntrinEmitter` —— 指令原子级 emitter(`pack_a/pack_b`=ldmatrix、`mma`=mxmem+MAC、`store`=stmatrix、`clear`=mxclracc),对齐 tilelang 的 `TensorCoreIntrinEmitter` | `tilelang/hexagon/hmx_intrin.py` | ✅ 设备验证 K-流式 |
| **运行时集成** | bridge(蹭 host 的 VTCM+HMX)、op registry(`tl_op_ctx`/`tl_dispatch`)、ggml 拦截、可嵌入 kernel emit、快路径 adapter | `src/tl_templates/hexagon/tl_{bridge,embed}.h`、`examples/hexagon/llama_cpp_integration/` | ✅ 完成 |
| **设备** | LFM2-1.2B 跑在 NPU 上,tilelang q4_0 matmul 在模型里 A/B | 见 §4 | ✅ 端到端验证,**追平** |

---

## 3. 逐层详解

### 3.1 Codegen:满寄存器 HVX 向量化(commit `287e95bb`)

**HVX 铁律:最窄 dtype 必须填满整个 128B 寄存器。** 混合 dtype 的循环(如 fp16 输出 + uint8
权重)必须在「最窄 dtype 是整数个 128B 寄存器」的宽度上向量化(uint8 需 ≥128 lane);否则会
越界读、在设备上 fault。

- `loop_vectorize.cc`:Hexagon 向量宽度从 128-bit(GPU 继承)改成 **1024-bit**,再要求最终
  向量长度是 `1024/最窄dtype位数` 的倍数;满足则升到该宽度,否则**安全标量化**。
- `common.h`:`vec_type` 用原生 clang `ext_vector` 实现(`a*b`、`a&mask` 变成真 HVX 指令,
  不是标量结构体循环),`aligned(1)`;`PrintType` 支持到 128 lane。
- 效果:任何 fp16 逐元素 DSL kernel 上满 HVX(干净的 `x*x` map:20ms → 1.4ms),零回归。

### 3.2 DSL kernel:全 DSL q4_0 matmul(commit `9db81aff`)

q4_0 的 nibble 解码是**纯 DSL 算术**,lower 成满宽 HVX(不需要手写反量化 intrinsic),和
`T.gemm`→HMX 融合成一个 kernel,权重留在 VTCM,不过 DDR。两条非显然的规则:

```python
q  = T.Cast("int16", qcm[j, n])                        # 先加宽到 int16
lo = (q & T.Cast("int16", 0xF)) - T.Cast("int16", 8)   # 再在 int16 上做 mask/shift
hi = (q >> T.Cast("int16", 4)) - T.Cast("int16", 8)    # int16 常量 → 不被提升成 int32(否则宽度被砍到 32)
```

权重预打包成**列主**(`qcm[K/2][N]`、`sc[K/32][N]`),让反量化的写是连续的(无 interleave/shuffle)。
量化/交织是**编解码**(computation),不是 `layout` —— 就像 CUDA 用 `mma` intrinsic 拼,而不是让
向量化器去神奇地加宽标量循环。结果:bit-exact,且**追平手写 HVX 模板**。

### 3.3 HMX 抽象:`HMXIntrinEmitter`(commit `76ada132`)

HMX 没有 mma、累加器不可见,所以 matmul 用**指令原子**围着那个隐式累加器拼,而不是黑盒
`T.gemm`。这是 tilelang `TensorCoreIntrinEmitter` 的 Hexagon 对应物:

| tensor core | `HMXIntrinEmitter` | 指令 |
|---|---|---|
| `ldmatrix_a/b`(shared→register fragment) | `pack_a/pack_b`(行主 VTCM→Crouton VTCM) | HVX `vshuff` |
| `mma`/`mma_atom` | `mma`(mxmem activation+weight load 对 = MAC) | `{activation=mxmem; weight=mxmem}` |
| `stmatrix` | `store`(`cvt`→Crouton→unpack) | `cvt.hf=acc; mxmem=cvt` |
| `T.clear(C_local)` | `clear` | `mxclracc` |

诚实的分歧:HMX 累加器**不可寻址、全核唯一**,所以 `clear/mma/store` 操作**隐式累加器**
(不像 CUDA 传 `C_local`);A/B 的「fragment」就是 `pack_*` 产出的 Crouton VTCM tile。用起来
就是 quickstart 的形状:`clear` → K-loop 里 `pack_b`/`mma` → `store`,反量化夹在 `pack_b` 和
`mma` 之间。方法内**建-返回一个 nested `@T.macro`** —— 必须这样,否则 Crouton scratch 的
buffer 使用对 VTCM liveness 分析不可见,会把 `alloc_shared` 叠在一起 → 静默算垃圾。

### 3.4 运行时集成(commits `3d570a15`、`f567ab15`)

llama.cpp 有自己的 DSP skel,tilelang kernel 要作为它图里的一个 op 跑,就得**嵌进它的 skel、
蹭它已经拿到的资源**(不能自己开 session / 重新 acquire VTCM/HMX):

- **bridge**(`tl_bridge.h`):把 tilelang runtime 的全局绑到 host 的 VTCM base + 骑 host 的
  HMX 锁。通用、干净,对任何嵌入 host skel 的 HMX/HVX kernel 都成立。
- **op registry**(`tl_embed.h`):`tl_op_ctx` + `tl_op_desc`(`matches`/`run`)+ `tl_dispatch`。
  op 在 skel 加载时自注册;host 的 stock kernel 只建一个 ctx 调 `tl_dispatch`,不认识这个 op。
  加一个 kernel = 一个自包含 `.cc`,不改任何 stock 函数。
- **拦截**:`matmul-ops.c` 的 `hmx_mm_2d_f32` 顶部 6 行,建 ctx 调 dispatch,`return -1` 回落
  到 stock。
- **可嵌入 kernel emit**:`k.get_kernel_source()` **本身就是可嵌入体**(一个 `extern "C"` 函数,
  用 `tl_vtcm_base()`,无 skel 壳)。`emit_embeddable.py` 吐它 + manifest。
- **快路径 adapter**(`tl_ggml_matmul.cc`):**tilelang 生成的满寄存器反量化** + **一次性缓存
  的 repack**(ggml 576B tile → 列主 `qcm`+`sc`)+ 复用的 HMX gemm。

---

## 4. 设备结果 —— LFM2-1.2B 在 NPU 上(本机 A/B)

从裸机建 llama.cpp `4fc4ec55` + ggml-hexagon 后端(Android NDK + Hexagon SDK 6.6,无 Docker),
接入集成,翻 `tl_mm_enabled` 开关(重编 skel、重新部署):

| 版本 | prefill(短) | prefill(长,warm) | decode | 输出 |
|---|---|---|---|---|
| stock | ~96–130 t/s | ~530 t/s | ~24–29 | 连贯 |
| tilelang,手写标量反量化 | 0.7 t/s | — | ~24 | 连贯 |
| **tilelang,生成的满寄存器反量化** | 4.3(repack 主导) | **537.7 t/s** | ~26 | 连贯 |

只翻一个开关、两边都连贯 → tilelang HMX 算子在 LFM2 前向里**既在跑、又算对**。**快路径追平**
后端手调的 q4_0 matmul(同长 prompt:537.7 vs 529.9 t/s)—— 正是预测的天花板。生成的满寄存器
反量化把手写标量的 0.7 拉到了打平;短 prompt 的 4.3 是**一次性 repack**(ggml tile→列主)记在
单次 prefill 上,部署时该放到 model-load。decode 不变:拦截在 HMX prefill 路径(`hmx_mm_2d_f32`),
解码(M=1)走 HVX GEMV,不被这个 HMX-only 的拦截碰到。

---

## 5. 硬道理(踩出来的关键发现)

1. **HVX 没有亚寄存器操作** —— 每条指令整个 128B 寄存器,所以混合 dtype 循环必须在最窄 dtype
   填满寄存器处向量化(uint8 需 ≥128 lane),否则越界 fault。这一条驱动了整个 codegen 修复。
2. **HMX 没有 mma、累加器不可寻址** —— load 即 MAC,进一个全核唯一的隐式累加器。所以 matmul
   用指令原子(`clear→pack→mma→store`)围着隐式累加器拼,是 `HMXIntrinEmitter` 而**不是** monolithic
   `T.gemm` —— 这样反量化才能夹进 MAC(K-流式)。
3. **Crouton pack 是固有的** —— 操作数进 `mxmem` 要 Crouton,但满寄存器反量化要行主,中间**必然**
   有一次行主→Crouton 的 HVX pack。`GemmHMX` 故意让操作数保持行主 + 运行时 HVX-pack,因为 Crouton
   layout 的 DSL 存储是标量(会把反量化打回标量)。
4. **tile op 必须 `@T.macro`** —— 普通 Python helper 里的 buffer 使用对 VTCM liveness 分析不可见,
   会把 `alloc_shared` 叠在一起 → **静默算垃圾**(不 crash)。emitter 方法内建-返回 nested macro。
5. **测 compute,别测单次墙钟** —— standalone kernel 的单次调用被 FastRPC marshal 主导;真 compute
   只在**权重常驻**(即模型里)才现出来。曾两次因 marshal 假象误判性能。
6. **单个 q4_0 matmul = 打平天花板** —— 后端本来就把 HVX 反量化融进了 HMX MAC。追平它(K-流式 /
   满寄存器反量化)到打平;**赢过 stock 要子图融合**(FFN/attention 一段留 VTCM)或更省字节的量化,
   而不是重写它的 q4_0 matmul。
7. **HVX 满寄存器 load 从 `malloc` 的 DDR 读要 ≥128B 对齐** —— 偏移是 128 倍数但 malloc 只给 16B →
   **静默算垃圾**(不 crash);快路径 cache 用 `memalign(256, …)`。用「同缓存、标量反量化」的诊断隔离。

---

## 6. 复现设备 A/B(裸机,无 Docker)

```bash
# 环境:Android NDK + Hexagon SDK 6.6(HEXAGON_TOOLS_ROOT 结尾到 .../19.0.07,不带 /Tools)
export ANDROID_NDK_ROOT=.../android-ndk-r25c HEXAGON_SDK_ROOT=.../Hexagon_SDK/6.6.0.0 \
       HEXAGON_TOOLS_ROOT=.../Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 4fc4ec55
cp docs/backend/snapdragon/CMakeUserPresets.json .
cmake --preset arm64-android-snapdragon-release -B build-snap -DGGML_OPENCL=OFF
cmake --build build-snap --target llama-cli htp-v79 -j$(nproc)          # baseline

TL=/path/to/tilelang-hexagon                                            # 接入
cp $TL/examples/hexagon/llama_cpp_integration/tl_ggml_matmul.cc ggml/src/ggml-hexagon/htp/
git apply $TL/examples/hexagon/llama_cpp_integration/ggml-hexagon.patch # -I 已指向 $TL
# A/B:改 tl_ggml_matmul.cc 里的 `int tl_mm_enabled` 0/1,然后:
cmake --build build-snap --target htp-v79 -j$(nproc)
adb push build-snap/ggml/src/ggml-hexagon/libggml-htp-v79.so /data/local/tmp/llamahtp/
adb shell "cd /data/local/tmp/llamahtp && LD_LIBRARY_PATH=. \
  ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp \
  ./llama-cli -m LFM2-1.2B-Q4_0.gguf --device HTP0 -ngl 99 -n 8 -st -p '<一段较长的 prompt>'"
```

坑:DSP 会跨次运行缓存已加载的 skel(含 adapter 的静态 cache),换 skel 后要 `pkill -9 -f llama`
再重拷强制 reload;短 prompt 冷启会被一次性 repack 主导,量真实的 matmul 速率要用长 prompt(warm)。

---

## 7. 下一步

- **赢过 stock → 子图融合** —— 用 registry 认领一段融合子图(FFN gate/up→SwiGLU→down 或
  attention),中间量留 VTCM,省掉后端逐 op 的 DDR 往返。这才是这套栈(codegen + emitter +
  bridge/registry)真正发力、能**超过** stock 的地方,而且复用上面全部。
- **快路径收尾(非阻塞)** —— repack 挪到 model-load(冷启不再被 repack 主导);为全部 shape
  生成按 shape 的全 DSL 融合 kernel(目前满寄存器**反量化**对 K/N 通用,全融合 DSL kernel 是定 shape)。
- **藏 K-流式的反量化**(task #29)—— whole-register 64-feature slice + 双缓冲,让反量化藏到 MAC 底下。

---

## 8. 文件索引

| | 位置 |
|---|---|
| 满寄存器 codegen | `src/transform/loop_vectorize.cc`、`src/tl_templates/hexagon/common.h`、`src/hexagon/codegen/codegen_hexagon.cc` |
| HMX / VTCM 运行时模板 | `src/tl_templates/hexagon/{hmx,vtcm,common,hvx_math}.h` |
| bridge / registry | `src/tl_templates/hexagon/{tl_bridge,tl_embed}.h` |
| HMX emitter | `tilelang/hexagon/hmx_intrin.py` |
| 全 DSL q4_0 matmul | `examples/hexagon/example_qmatmul.py`、`example_qmatmul_kstream.py` |
| llama.cpp 集成(adapter、patch、可嵌入 emit) | `examples/hexagon/llama_cpp_integration/` |
| 细节文档 | `docs/{edge_tilelang_vs_cuda,hexagon_dsl_kernels,llama_cpp_integration,lfm2_1.2b_compute_graph,llama_cpp_reproduce,llama_cpp_inference_flow}.md` |
