# 复现:LFM2 在 Hexagon NPU 上跑 + 换入 tilelang 算子(llama.cpp 穿刺版)

从零到「LFM2 在 NPU 上吐词」再到「把一个 tilelang q4_0 matmul 换进模型、A/B 对比」的**全部命令**。
本轮实测于 **Snapdragon 8 Elite(Hexagon v79)**,主机为无 root 的 Linux。所有命令都真跑过。

> 说明:这是一个**正确性穿刺**(证明链路通、算子对),tilelang 路径故意没优化(标量反量化 + M padding),
> 慢是预期的。机制/反思见 `docs/llama_cpp_integration.md`,推理链见 `docs/llama_cpp_inference_flow.md`。

---

## 0. 前置(按你的机器改路径)

```bash
# —— 主机工具(版本是本轮验证过的组合)——
export NDK=/path/to/android-ndk-r25c                       # Android NDK r25c
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.6.0.0       # Hexagon SDK 6.6.0.0
export HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.07   # ⚠ 结尾到 19.0.07,不要带 /Tools
export ANDROID_NDK_ROOT=$NDK
export PATH=$HOME/.local/cmake/bin:$PATH                   # cmake ≥3.22 + ninja
export TL=/path/to/tilelang-hexagon                        # 本仓库

# —— 设备 —— Snapdragon 8 Gen3/Elite 一类,已 `adb devices` 授权;确认 DSP 架构(v73/75/79/81)
adb devices
```

需要:Android NDK、Hexagon SDK(社区版 6.6+)、CMake+Ninja、一台已授权的 Snapdragon 设备。
不需要:tilelang 本身(穿刺只用到已提交的 `.cc`/`.patch`/头文件)、OpenCL SDK。

---

## 1. 编译 llama.cpp + Hexagon 后端

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
# 集成 patch 是对 base 4fc4ec5 生成的,实测也能干净打在当前 upstream HEAD(ed8c261)上
# (matmul-ops.c / CMakeLists.txt 没漂移)。第 4 步打 patch 前会先 `git apply --check` 验证。

cp docs/backend/snapdragon/CMakeUserPresets.json .
cmake --preset arm64-android-snapdragon-release -B build-snap -DGGML_OPENCL=OFF   # OpenCL 关掉,省掉一个 SDK 依赖

# 只 build 我们要的 target:llama-cli + DSP skel。
# （不要 `cmake --build build-snap` 全量 —— 它会去编 llama-ui-embed,那个用 host 编译器找不到 <algorithm> 会失败）
ninja -C build-snap llama-cli
ninja -C build-snap htp-v79               # DSP skel;换 arch 就 htp-v73 / htp-v75 / htp-v81
```

产物:
```
build-snap/bin/llama-cli, libggml*.so, libllama*.so
build-snap/ggml/src/ggml-hexagon/libggml-htp-v79.so         # cDSP skel
```

---

## 2. 拿模型(q4_0)

```bash
# 1.2B(小,先跑通):
wget https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF/resolve/main/LFM2-1.2B-Q4_0.gguf
# 8B-A1B(MoE,真正目标,~4.5GB):
# wget https://huggingface.co/LiquidAI/LFM2-8B-A1B-GGUF/resolve/main/LFM2-8B-A1B-Q4_0.gguf
```
> 后端接受 q4_0（会 repack 上 HMX）。别放到 tmpfs（8B 放不下）。

---

## 3. 部署 + 跑(stock,先证明 LFM2 上 NPU)

```bash
D=/data/local/tmp/llamahtp
adb shell "mkdir -p $D"
adb push build-snap/bin/. "$D/"
adb push build-snap/ggml/src/ggml-hexagon/libggml-htp-v79.so "$D/"
adb push "$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" "$D/"
adb push LFM2-1.2B-Q4_0.gguf "$D/"

adb shell "cd $D && LD_LIBRARY_PATH=$D \
  ADSP_LIBRARY_PATH=$D:/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp \
  ./llama-cli -m LFM2-1.2B-Q4_0.gguf --device HTP0 -ngl 99 -n 24 -st -p 'The capital of France is'"
```
预期:`The capital of France is Paris…`,`Generation: ~34 t/s`(vs CPU ~1 t/s)。

- **`-st` 必须加**:这个 build 没有 `-no-cnv`,不加 `-st` 会卡在对话模式空转。
- **8B-A1B**:加 `GGML_HEXAGON_NDEV=4 ... --no-mmap`(把 4.5GB 摊到 4 个 DSP session,绕 32-bit cDSP 的 4GB 地址空间),约 28 t/s。

---

## 4. 换入 tilelang 算子(穿刺的核心)+ A/B

```bash
# a) 放入集成 TU + 打 patch(patch 改 matmul-ops.c 拦截点 + CMakeLists）
P="$TL/examples/hexagon/llama_cpp_integration"
cp "$P/tl_ggml_matmul.cc" ggml/src/ggml-hexagon/htp/
git apply --check "$P/ggml-hexagon.patch" && git apply --3way "$P/ggml-hexagon.patch"
#   万一 upstream 漂移打不上:patch 只有 ~20 行,照它手改(matmul-ops.c 顶部加拦截 + CMakeLists 加 .cc/-I/-Wno-unused-function)

# b) patch 里的 include 路径是绝对路径,改成你的 $TL —— 编辑 htp/CMakeLists.txt 里的 include_directories：
#      /home/xwh/tilelang-hexagon/src            → $TL/src
#      /home/xwh/tilelang-hexagon/src/tl_templates/hexagon → $TL/src/tl_templates/hexagon
#    （tl_embed.h / tl_bridge.h 已在 $TL/src/tl_templates/hexagon/,不用拷）

# c) A/B 开关:htp/tl_ggml_matmul.cc 里 `int tl_mm_enabled = 0;`  0=stock,1=tilelang。先设 1 验证：
sed -i 's/^int tl_mm_enabled = 0;/int tl_mm_enabled = 1;/' ggml/src/ggml-hexagon/htp/tl_ggml_matmul.cc

# d) 重编 skel + 推
ninja -C build-snap htp-v79
adb push build-snap/ggml/src/ggml-hexagon/libggml-htp-v79.so "$D/"

# e) 跑（-n 6 就够，tilelang 路径慢）
adb shell "cd $D && LD_LIBRARY_PATH=$D \
  ADSP_LIBRARY_PATH=$D:/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp \
  ./llama-cli -m LFM2-1.2B-Q4_0.gguf --device HTP0 -ngl 99 -n 6 -st -p 'The capital of France is'"
```

**判读 A/B(翻转 `tl_mm_enabled` 0↔1,重编 htp-v79,重跑）**:
- `=0`(stock):`Prompt: ~156 t/s`
- `=1`(tilelang):`Prompt: ~0.8 t/s`,**输出仍连贯**
→ 一个 flag 让 prefill 差 ~195× 且两边都连贯 = 证明 tilelang 算子**真的在跑、且结果对**。
(只有 prefill 的 HMX-path matmul 被接管;decode 的 M=1 走后端 HVX GEMV,本拦截点不碰,所以 decode 仍 ~30 t/s。)

---

## 5. 代码示例(接一个 tilelang 算子长什么样)

分两侧:**作者侧**(tilelang DSL 写 kernel)和**集成侧**(把它接进 llama.cpp 的三小段胶水)。
完整文件在 `examples/hexagon/llama_cpp_integration/tl_ggml_matmul.cc` + `src/tl_templates/hexagon/{tl_embed,tl_bridge}.h`。

### 5.1 作者侧:用 tilelang DSL 写 q4_0 反量化 kernel

跟写 GPU 的 tilelang 一样——`T.alloc_shared`(落 VTCM)+ `T.copy` + 逐元素循环(codegen 会向量化到 HVX):

```python
import tilelang, tilelang.language as T

def make_dequant_q4_0(N, K, BLK=32):
    KB = K // BLK
    @T.prim_func
    def dequant(Q: T.Tensor((N, K // 2), "uint8"),    # 打包的 4-bit nibble
                S: T.Tensor((N, KB), "float16"),       # 每 32-block 一个 fp16 尺度
                W: T.Tensor((N, K), "float16")):       # 反量化输出
        with T.Kernel(1, threads=1) as _:
            Q_sh = T.alloc_shared((N, K // 2), "uint8")
            S_sh = T.alloc_shared((N, KB), "float16")
            W_sh = T.alloc_shared((N, K), "float16")
            T.copy(Q, Q_sh); T.copy(S, S_sh)
            for i in T.serial(N):
                for b in T.serial(KB):
                    for j in T.serial(16):             # 一个字节 = 两个权重(低/高 nibble)
                        q = T.cast(Q_sh[i, b * 16 + j], "int32")
                        d = S_sh[i, b]
                        W_sh[i, b * 32 + j]      = (T.cast(q & 0xF, "float16")      - T.cast(8, "float16")) * d
                        W_sh[i, b * 32 + j + 16] = (T.cast((q >> 4) & 0xF, "float16") - T.cast(8, "float16")) * d
            T.copy(W_sh, W)
    return dequant

# 编译 → 上设备 → 对拍 numpy(独立验证,不进 llama.cpp)。M2 里就是这样把每块都验过的:
kernel = tilelang.compile(make_dequant_q4_0(32, 256), out_idx=[2], target="hexagon")
# W = kernel(Q, S)  → 与 (nibble-8)*scale 逐位相等
```
HMX matmul 同理:`T.copy → T.gemm(clear_accum=True) → T.copy`,`T.gemm` 被 codegen 下降到 HMX
(见 `examples/hexagon/example_matmul.py`)。

### 5.2 集成侧:三小段胶水把它接进 llama.cpp

**① 算子描述符(谓词 + 实现),`__attribute__((constructor))` 自注册——不改任何 stock 函数:**
```cpp
// htp/tl_ggml_matmul.cc
static int tl_mm_matches(const struct tl_op_ctx *o) {           // 「这个 op 是我的吗」
  return o->kind == TL_OP_MATMUL && o->weight_type == HTP_TYPE_Q4_0 && !o->has_bias &&
         o->m <= 32 && (o->k % 32) == 0 && (o->n % 32) == 0;
}
static int tl_mm_run(const struct tl_op_ctx *o) {              // 「就由我算」
  HAP_compute_res_hmx_lock(o->vtcm_rctx);       // 此拦截点 HMX 还没锁,自己锁
  tl_bridge_enter(o->vtcm_base, o->vtcm_size);  // 绑定到 host 的 VTCM,蹭它的 HMX
  for (int n0 = 0; n0 < o->n; n0 += NC) {
    tl_dequant_q4_0_chunk(o->weight, Wf, o->k, n0, nc, nkt);   // repack tile → 行主 fp16
    tl_hexagon_hmx_gemm(Cf, Af, Wf, 32, nc, o->k, 0, 0);       // tilelang 的 HMX gemm
    /* 写回 o->dst ... */
  }
  tl_bridge_exit();
  HAP_compute_res_hmx_unlock(o->vtcm_rctx);
  return 0;                                     // 返回 0=我接了;返回 -1=退给 stock
}
static const struct tl_op_desc tl_mm_desc = { "q4_0_matmul", tl_mm_matches, tl_mm_run };
__attribute__((constructor)) static void reg(void) { tl_register_op(&tl_mm_desc); }
```

**② host 拦截点(patch 加在 `hmx_mm_2d_f32` 顶部)——只建 ctx + 派发,从不点名任何算子:**
```c
// htp/matmul-ops.c
struct tl_op_ctx octx = { ctx->vtcm_base, ctx->vtcm_size, ctx->vtcm_rctx,
                          TL_OP_MATMUL, weight_type, (src2 != NULL),
                          dst, activation, weight, m, k, n, act_stride, dst_stride };
if (tl_dispatch(&octx) == 0) return 0;          // 有 tilelang 算子接了就返回,否则往下走 stock
```

**③ bridge(蹭 host 的 VTCM + HMX,不自己申请)——这是端侧和 CUDA 的关键差异所在:**
```c
// src/tl_templates/hexagon/tl_bridge.h
static inline void tl_bridge_enter(void *vtcm_base, unsigned vtcm_size) {
  tl_vtcm_base_ptr = (uint8_t *)vtcm_base;       // tl_vtcm_acquire() 变 no-op → 用 host 的 VTCM
  tl_vtcm_total    = vtcm_size;
  tl_hmx_fill_unit_scales((uint32_t *)tl_vtcm_base_ptr);  // 我们的 HMX 输出尺度 @ base+0
  tl_hmx_inited    = 1;                           // 蹭调用方已持有的 HMX 锁,不做第二次 acquire
}
```

**加第二个算子** = 再写一个 `.cc`,里面一个 `tl_op_desc` + `__attribute__((constructor))` 自注册,别的都不动。
(把注册表上提到 `supports_op`/`graph_compute` 后,连「后端本来不支持的新 op」也能这样加——见 `docs/llama_cpp_integration.md` 的路线。)

---

## 6. 排查 / 抓数据(全是运行期环境变量,不用重编)

```bash
# 每个算子落 CPU 还是 HTP（ggml 核心 scheduler 的真实分配）+ hexagon 的 supports-op 明细：
GGML_SCHED_DEBUG=2 GGML_HEXAGON_VERBOSE=1 ./llama-cli ... -v   # 看 `node #N (OP): name [CPU|HTP0]`
# 每算子耗时：
GGML_HEXAGON_PROFILE=1 ./llama-cli ... -v                     # `profile-op … usec N`
# 强制某些算子回 CPU（正则）：
GGML_HEXAGON_OPFILTER='.*MUL_MAT.*' ./llama-cli ...
# 大模型多 session：
GGML_HEXAGON_NDEV=4 ./llama-cli ... --no-mmap
```
LFM2-1.2B 实测算子分布见 `examples/hexagon/llama_cpp_integration/lfm2_1.2b_op_split.md`(352/354 在 NPU,
只有 q6_K 的 embedding 查表 + LM head 在 CPU)。

---

## 7. 坑(踩过的,省得别人再踩)

- **`HEXAGON_TOOLS_ROOT` 结尾到 `19.0.07`,别带 `/Tools`** —— htp 的 ExternalProject 自己会拼 `/Tools/bin`,带了就双 Tools 找不到 `hexagon-clang`。
- **别 `cmake --build` 全量** —— `llama-ui-embed`(host 编译)会 `fatal error: 'algorithm' file not found`。只 build `llama-cli` + `htp-vNN` 两个 target。
- **`-DGGML_OPENCL=OFF`** —— 否则要 OpenCL SDK;我们只要 CPU+Hexagon 后端。
- **跑必须 `-st`** —— 这个 build 没有 `-no-cnv`;不加会卡对话模式。
- **scratch 若是 tmpfs(小),别把模型/build 放进去** —— 4.5GB 的 8B 放不下,撑爆了连 `adb` 都会 "Disk quota exceeded"。
- **设备侧 `FARF` 默认到不了 logcat**(要 `<exe>.farf` 配置)——设备端正确性验证用 abort/返回码当信号,别指望 printf。
- **改了 DSP 侧源码后重编**:直接 `ninja -C build-snap htp-v79` 即可;若曾手删过 `htp-vNN-prefix/src/htp-vNN-build`,要 `rm -rf htp-vNN-prefix` 让 ExternalProject 重 configure(只删 build 目录会因 `cd` 不到而报错)。
- **HMX 只在你自己的 PD 里可用**:tilelang 算子必须编进 host 的 skel(见 bridge),不能做成独立 skel;QNN 的 external op-package 拿不到 HMX(实测)。
