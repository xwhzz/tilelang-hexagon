# tilelang 算子对接示例:一个 q4_0 matmul 从 DSL 跑进 llama.cpp

一个**可参考的完整闭环** —— 用 tilelang DSL 写一个量化 matmul 算子,编译成可嵌入的 C,
对接进 llama.cpp 的 ggml-hexagon 后端,在设备上 A/B。这是当前已 device-validated 的路径
(prefill q4_0 matmul,达 parity)。可运行代码:`examples/hexagon/example_qmatmul.py`
(DSL 算子)+ `examples/hexagon/llama_cpp_integration/`(对接全套)。

> 说明:这条路对接的是 **prefill 的 HMX matmul**(M≤32),已追平 backend。基于本轮 profiling
> 的战略,下一个算子应对接 **decode 的量化 GEMV / 短卷积块融合**(见 `tilelang_npu_slides.html`
> 第 7 页)。对接**机制**完全一样,换的是算子本身。

---

## 全景:6 步

```
① DSL 算子        →  ② 编译+抽取        →  ③ 发射 .c + manifest
   @T.prim_func       get_kernel_source()    kernel_*.c + .manifest.json
        │                                          │
        ▼                                          ▼
④ host adapter    →  ⑤ bridge(骑资源)   →  ⑥ patch 拦截 + 部署 A/B
   tl_op_desc 注册     tl_bridge_enter        hmx_mm 顶部 tl_dispatch
```

---

## ① 用 tilelang DSL 写算子

反量化就是普通 DSL 算术(codegen 自动 lower 成整宽 HVX),`T.gemm` lower 到 HMX。
反量化后的权重留在 VTCM(`alloc_shared`),不回 DDR。

```python
import tilelang, tilelang.language as T

def make_qmatmul(M, N, K):
    KH = K // 2
    @T.prim_func
    def qmatmul(A:   T.Tensor((M, K),  "float16"),   # 激活
                qcm: T.Tensor((KH, N), "uint8"),     # q4_0 权重,column-major 预打包
                scb: T.Tensor((K, N),  "float16"),   # per-(k,n) scale
                C:   T.Tensor((M, N),  "float16")):
        with T.Kernel(1, threads=1) as _:
            A_sh = T.alloc_shared((M, K), "float16")
            B_sh = T.alloc_shared((K, N), "float16")  # 反量化后的权重,留 VTCM
            C_sh = T.alloc_shared((M, N), "float16")
            T.copy(A, A_sh)
            for j in T.serial(KH):                    # ── DSL q4_0 dequant → 整宽 HVX ──
                for no in T.serial(N // 128):
                    for ni in T.vectorized(128):      # 铁律①:填满 128B 寄存器
                        n  = no * 128 + ni
                        q  = T.Cast("int16", qcm[j, n])                    # 铁律②:先加宽
                        lo = (q & T.Cast("int16", 0xF)) - T.Cast("int16", 8)
                        hi = (q >> T.Cast("int16", 4)) - T.Cast("int16", 8)
                        B_sh[2*j,   n] = T.Cast("float16", lo) * scb[2*j,   n]
                        B_sh[2*j+1, n] = T.Cast("float16", hi) * scb[2*j+1, n]
            T.gemm(A_sh, B_sh, C_sh, clear_accum=True)  # → HMX(必须 clear_accum)
            T.copy(C_sh, C)
    return qmatmul
```

**三条 HVX 铁律**(否则设备上静默出错/变慢,不是崩溃):
1. **最窄 dtype 必须填满整个 128B 寄存器** —— uint8 需 ≥128 lane,用 `T.vectorized(128)`。
2. **bitwise 前先加宽到 int16** —— uint8 在半寄存器宽度做位运算会 over-read 触发 fault。
3. **权重预打包 column-major**(`qcm[K/2][N]`)—— 让 dequant 的存储连续,不用 shuffle。

**两层 DSL surface**:上面是简单版(一个 `T.gemm`)。要让 dequant **藏在 MAC 下**(K-streaming),
用 `HMXIntrinEmitter` 手驱 K-loop(`begin/pack_a/clear/pack_b/mma/store/end`),见
`example_qmatmul_kstream.py`。

---

## ② 编译,抽取可嵌入的 C body

```python
k = tilelang.compile(make_qmatmul(32, 128, 2048), out_idx=[3], target="hexagon")
src = k.get_kernel_source()      # ← 关键:这本身就是可嵌入的 body
```

**关键事实**:`get_kernel_source()` 产出的就是一个裸的 `extern "C"` 函数,内部调用
`tl_vtcm_base()` + `tl_hexagon_hmx_gemm` —— **没有 `_skel.so` / session-acquire 包装**
(那层只在独立部署时才加)。嵌入只需要这段源码 + bridge 头文件。

```c
extern "C" int32_t qmatmul_compact_kernel(half* A, uint8_t* qcm, half* sc, half* C) {
    /* ... 生成的 HVX dequant + tl_hexagon_hmx_gemm ... 用 tl_vtcm_base() 拿 VTCM ... */
}
```

`emit_embeddable.py` 把它写成 `kernel_qmatmul_compact_32x128x2048.c` + 一份
`.manifest.json`(入口签名、op=matmul/dtype=q4_0、M/N/K、VTCM 用量、权重格式契约)。
为了常驻,scale 用**紧凑的** `sc[K/32][N]`(展开的 `scb[K][N]` 是权重的 4×,放不下),在 kernel 内展开。

---

## ③–④ host adapter:注册进 op registry

`tl_ggml_matmul.cc`(放进 `ggml/src/ggml-hexagon/htp/`)提供一个自注册的 op 描述符。
这是**通用的 registry 分发**,不是硬编码的 per-op `if`:

```c
static struct tl_op_desc TL_Q4_MM = { "q4_0_matmul", tl_mm_matches, tl_mm_run };
__attribute__((constructor)) static void reg(void){ tl_register_op(&TL_Q4_MM); }

// 门控:只接收 HMX 能处理的 prefill q4_0 matmul
static int tl_mm_matches(const struct tl_op_ctx* o){
    return o->kind==TL_OP_MATMUL && o->weight_type==HTP_TYPE_Q4_0
        && !o->has_bias && o->vtcm_base && o->m<=32 && o->k%32==0 && o->n%128==0;
}
static int tl_mm_run(const struct tl_op_ctx* o){
    tl_bridge_enter(o->vtcm_base, o->vtcm_size);   // 骑 host 资源(见 ⑤)
    /* 一次性 repack: ggml tiled q4_0 → qcm/sc(缓存);再按 N-chunk 调 kernel */
    tl_bridge_exit();
    return 0;                                       // 返回 -1 = 声明不处理 → 回退 backend
}
```

---

## ⑤ bridge:骑 host 已获取的 VTCM + HMX(Mode B)

**核心设计**。tilelang 算子作为**源码**co-compile 进 host 的 `libggml-htp-v79.so`,
**复用** host 已经 acquire 的 VTCM 区 + HMX 锁,**不自己开 session** —— 这是能嵌进别人 skel 的前提。

```c
void tl_bridge_enter(void* vtcm_base, size_t vtcm_size){
    tl_vtcm_base_ptr = vtcm_base;    // tl_vtcm_acquire() 变成 no-op,用 host 的 VTCM
    tl_hmx_write_unit_scale_tile();  // 在 base+0 填 HMX unit-scale tile
    tl_hmx_inited = 1;               // 骑 host 的 HMX 锁,不重新 lock
}
```

`tl_bridge.h` 只有 ~6 行,是整个对接里最可复用、最可移植的部分。

---

## ⑥ patch 拦截点 + 部署 A/B

`ggml-hexagon.patch` 两处改动:

**(a) 在 `matmul-ops.c` 的 `hmx_mm_2d_f32(...)` 顶部**插入分发(命中就走 tilelang,否则 fall through):
```c
struct tl_op_ctx octx = { ctx->vtcm_base, ctx->vtcm_size, ctx->vtcm_rctx,
                          TL_OP_MATMUL, weight_type, (src2!=NULL),
                          dst, activation, weight, m, k, n, act_stride, dst_stride };
if (tl_dispatch(&octx) == 0) return 0;   // 命中 tilelang → 返回;-1 → 继续原内核
```
**(b) `CMakeLists.txt`**:加 `tl_ggml_matmul.cc`、`-I` 指向 tilelang 模板、`-Wno-unused-function`。

部署 + A/B(主开关 `int tl_mm_enabled` 在 `tl_ggml_matmul.cc` 里,0/1 切换后重编):
```bash
cmake --build build-snap --target htp-v79 -j$(nproc)
adb push build-snap/ggml/src/ggml-hexagon/libggml-htp-v79.so /data/local/tmp/llamahtp/
adb shell "cd /data/local/tmp/llamahtp && LD_LIBRARY_PATH=. \
  ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp \
  ./llama-cli -m LFM2-1.2B-Q4_0.gguf --device HTP0 -ngl 99 -n 8 -st -p 'The capital of France is'"
```
翻 `tl_mm_enabled`,两个 arm 都输出连贯 → tilelang 算子**在模型前向里既激活又正确**。

---

## 安全边界与踩过的坑

| | |
|---|---|
| **回退契约** | `run()` 返回 `-1` 就回退 backend 原内核 → tilelang 路径可以**只覆盖部分形状**,不影响模型正确性。 |
| **必须 `clear_accum=True`** | HMX 累加器要先 `mxclracc`,否则 NaN。 |
| **VTCM tile 必须 `@T.macro`** | Emitter 方法要 build+return 嵌套 `@T.macro`,否则 VTCM liveness pass 看不到 scratch buffer 的使用 → 别名其它 `alloc_shared` → 静默出错。 |
| **对齐** | HVX 整寄存器从 `malloc` 的 DDR 加载需 ≥128B 对齐(`memalign(256,…)`),否则静默垃圾。 |
| **测 compute 不是单次墙钟** | 单次 `kernel()` 被 FastRPC input marshal 主导,曾两次误判性能。要么多次 amortize,要么用常驻权重(即在模型里测)。 |
| **单算子是 parity 天花板** | backend 已把 q4_0 dequant 融进 HMX MAC。追平它(K-streaming)到 parity;**要超越得靠子图融合或更少字节的量化**。 |

---

## 下一个算子该对接什么(基于本轮 profiling)

对接**机制**完全复用上面 6 步,换的是 ① 的 DSL 算子:

1. **decode 量化 GEMV**(FFN gate/up/down,M=1)—— 69% 的 decode 时间,HMX 闲置,backend 弱项。`tl_mm_matches` 的门控改成接收 `m==1`,DSL 写高效 HVX GEMV(或探索 HMX 用法)。
2. **短卷积块融合**(`in_proj → SSM_CONV → out_proj`)—— 唯一没被 ggml 融合的子图,中间量留 VTCM。这需要一个新的 `tl_op_desc`(不是 matmul 门控,而是匹配这个子图模式)。
3. **lm_head**(已验证 +80%)—— 本轮已用「改上限 + 重量化」让 backend 自己的内核跑;若要 tilelang 版,就是一个 N=65536 的 q4_0 GEMV。
