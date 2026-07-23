# 交接文档:tilelang × Hexagon —— 找到 8-bit decode 的正确抽象,实现一个算子进真实推理

> **2026-07-15 复核更新:** 本文保留为调查过程记录，但其中“Q8 decode 缺 DMA/双缓冲”、
> “26.6 tok/s 是公平的全 HTP 基线”以及“模型大小 × tok/s 就是 DDR 实际带宽”三个前提已经被
> 设备 A/B 和源码阅读推翻。ggml 的 Q8 路径已经使用 6-worker、VTCM、2D DMA 和 2–16 路预取；
> 原始 Q8 基线还把 65536-row `lm_head` 拒绝到 CPU。放宽限制后的公平基线是
> **35.68 ± 0.01 tok/s**，TileLang Q8 atom 模型集成为 **35.67 ± 0.37 tok/s**，对应 stock
> A/B 为 **35.46 ± 0.17 tok/s**（性能持平，非加速）。新的设计、实现和复现记录见
> [`tilelang_hexagon_q8_design.md`](tilelang_hexagon_q8_design.md)。下文未逐段改写的旧结论不能再作为
> backend 重构依据。

> **2026-07-16 Q4 修复状态:** Q4_0 现在不是 opaque C dequant 或 `T.gemm` 包装。checked-in
> K=2048/8192 kernel 在 TileLang 中显式重建 FP16 scale、拆 low/high nibble、减 8、乘 scale，
> 并通过 `T.Layout` 直接写入 HMX weight Crouton，再调用显式 HMX atoms。原来的编译 OOM 来自
> `<64 x i32>` nibble 向量触发 Hexagon SelectionDAG 爆炸；改为 low/high 两个 32-lane 整数阶段，
> FP16 后再 interleave，两个 shape 的 v79 编译峰值约 113 MB。设备单算子相对误差分别为
> `3.5e-4`/`3.4e-4`。llama.cpp 的真实 LFM2 Q4_0 `pp32` 已命中 adapter 且稳定，但当前只有
> **2.85 t/s**，stock 为 **590.33 t/s**：瓶颈是单线程 dequant 和过细的固定 `M32/N128`
> 调度，不是 HMX。Q4 feature 必须继续默认关闭，下一步应把 HVX dequant atom 和 6-worker
> scheduling 暴露给 TileLang，而不是再往上封装成新的 monolithic tileop。

**给接手的 agent。** 这份文档是一次深度性能剖析 session 的产物 + 一个明确的下一步任务的
briefing。目标读者是一个要继续"反思 tilelang↔Hexagon 的对接设计、理解硬件、找到适合的抽象、
用 tilelang 实现一个 **8-bit** 量化算子替换进 LFM2 真实推理"的 agent。

> **约束(用户明确):只考虑 8-bit(Q8_0 / w8a16)量化**,以便和 nexa AI 的数据严格对齐。
> 之前的 parity 工作是在 q4_0(4-bit)上做的 —— 换到 8-bit 是这次的方向。

标注约定:**[实测]** = 本 session 在设备上测到的;**[源码]** = 读代码确认的;**[推理]** =
从实测+原理推出的可靠结论;**[存疑]** = 未证实的推测,接手方需验证。

---

## 0. TL;DR —— 三句话

1. **战场在 decode,不在 prefill。** prefill 的 HMX matmul 已被 tilelang 追平(parity),那是
   backend 的强项;真正的机会在 decode —— 同精度(8-bit)下 llama.cpp 比 nexa 慢 **2×** [实测]。
2. **decode 是权重带宽 bound,不是算力 bound。** llama Q8_0 decode 只跑到 DDR 峰值的 ~40% [实测/推理];
   HMX 在 decode 时闲置是**正确的**(矩阵引擎对 M=1 GEMV 无用),别再想"把 decode 塞进 HMX"。
   **正确的抽象是"VTCM 流式 + DMA 双缓冲 + worker-pool 并行的反量化-点积",不是矩阵引擎抽象。**
3. **下一个算子 = 一个 Q8_0 decode GEMV**(建议先做 lm_head 或 FFN down 的独立 GEMV),
   目标是掌握"逼近峰值带宽的权重流式"这套抽象,再谈子图融合。

---

## 1. 任务(接手方要做的)

> 用 tilelang 实现**一个 8-bit(Q8_0)算子**,替换进 LFM2-1.2B 的**真实 decode 推理**,
> 测出对 backend HVX GEMV 的提升,向 nexa 的 8-bit decode(69.5 tok/s)靠拢。

但在写代码之前,用户要求**先想清楚三件事**(这份文档给了起点,但需要接手方深入):

1. **理清 tilelang 怎么对接 hexagon backend** —— 现有机制见 §4,但它只拦 prefill;decode 要找新的对接点。
2. **反思现有设计与实现,找到适合 8-bit decode 的抽象** —— 我的当前判断见 §6,但有 4 个必须先验证的开放问题(§7)。
3. **理解硬件架构** —— §2。特别是"HMX 是 fp16 矩阵引擎、decode 是 M=1 GEMV"这个根本张力。

---

## 2. 硬件架构(必须先吃透)

**设备** [实测]:OnePlus PJZ110 · 骁龙 8 Elite(SoC `SM8750`,平台 `sun`)· Android 15 · Hexagon **v79** NPU。

| 单元 | 规格 | 对 decode 的含义 |
|---|---|---|
| **HVX**(向量) | **1024-bit / 128 字节**寄存器 × **6 硬件线程** | decode GEMV 跑在这。**没有子寄存器操作** —— 每条指令都是整个 128B 寄存器(见 §5 铁律)。 |
| **HMX**(矩阵) | **1 个** fp16 矩阵引擎,32×32 **Crouton** tile,单个隐形累加器 | 为**批量** matmul(M≥32,prefill)而生。**对 M=1 GEMV 无用**(浪费 32 行 tile)。 |
| **VTCM** | **8 MB** 软件管理 SRAM | 权重从 DDR 流式进来的落脚点。**没有自动缓存**,要显式 DMA + 双缓冲。 |
| **DDR** | LPDDR5X,峰值 **~77–85 GB/s** [参考值,需接手方确认 SM8750 具体值] | decode 每 token 要把**全部权重**流一遍 → decode 的天花板 = DDR 峰值 / 权重字节。 |

**关键理解 —— decode 的物理本质:** LLM decode 是 M=1 的 GEMV(矩阵×向量),每个权重只用一次
(算术强度 = 1)→ **访存 bound,不是算力 bound**。所以:
- **HMX 闲置是对的**:矩阵引擎解决不了访存 bound 的问题,它的价值在权重复用(prefill 的 M=1024)。
- decode 的上限 = `DDR峰值 / 权重字节`。Q8_0 权重 1.19 GB,峰值 ~80 GB/s → 理论 ~67 tok/s。
  **nexa 实测 69.5 ≈ 打满了带宽;llama Q8_0 只有 24-35 = ~40% 峰值。** 差距就在这。

**HMX 的 dtype** [源码]:tilelang 的 HMX runtime(`src/tl_templates/hexagon/hmx.h`)**只有 `__fp16` 路径**
(`tl_hexagon_hmx_gemm(__fp16*, ...)`)。Hexagon HMX 硬件**是否支持 int8 矩阵模式我没查完**
(被打断)—— 这是 §7 的开放问题 #1。但即便支持,对 M=1 GEMV 也无用(见上)。

---

## 3. 本 session 的性能剖析发现(证据基础)

完整报告:`docs/lfm2_hexagon_report/`(中文,含可复现脚本 + 原始 log)、`docs/hexagon_lfm2_perf.md`、
`docs/hexagon_profiling.md`。交互式:`docs/hexagon_timeline.html`、`docs/lfm2_perfetto_trace.json`。

**性能 @1024-token 输入** [实测,`llama-bench -p 1024 -n 128 -r 3`]:

| 运行时 | 权重 | prefill | decode | decode @d1024 |
|---|---|---:|---:|---:|
| **nexa / GenieX**(QNN,官方报告) | 8-bit | 3618 | **69.5** | 69.5 |
| llama.cpp | **Q8_0(8-bit)** | 3272 | 26.6 | 24.1 |
| llama.cpp | Q4_0(4-bit) | 3175 | 35.6 | 33.2 |
| llama.cpp | F16 | 2174 | 22.1 | — |

- **decode 受权重带宽限制** [实测]:Q4_0 > Q8_0 > F16(字节越少越快)。所以在 llama 里 Q4_0 才是最快的,
  但**用户要求对齐 nexa,所以我们工作在 Q8_0**。
- **同精度(8-bit)nexa 快 2.6×**(69.5 vs 24.1)[实测/对齐条件]。差距是运行时/内核效率,不是量化。
- **框架 overhead 很小,不是瓶颈** [实测]:纯框架路径(`GGML_HEXAGON_OPSTAGE=1` 只排队不计算)
  只有 3.5ms/token(13%)。别去优化框架 —— 瓶颈是真实的权重流式 + GEMV。

**完整前向剖析(一个 decode token)** [实测,`GGML_HEXAGON_PROFILE=2` + 8 个 PMU 计数器]:
- 227 个 NPU 算子,~13.5–14.7 ms DSP 计算,16 层惊人均匀(~0.83ms/层)。
- **FFN 占 69%**(gate+up 46% + down 23%),短卷积 15%,注意力 11%。
- **引擎归属**(用 `HVX_ACTIVE` PMU 事件 `0x100` = ggml `pmu[2]` 验证):HMX **5.2%** / HVX **~95%** / scalar ≈0。
- decode 段**没有一个算子走 HMX**(kparams 全 `hvx-tiled`/`----`)。

**本 session 已拿到的实战收益(非 tilelang,但把差距压小了)** [实测]:
- **lm_head 搬回 NPU:decode +80%**(Q4_0 @d1024 33.2→54.8)。ggml-hexagon 有一句
  `if (src0->ne[1] > 32768) return false; // hardcoded limit to refuse the lm-head for now`
  (`ggml/src/ggml-hexagon/ggml-hexagon.cpp:2797`)把 65536 词表的 lm_head 赶去 CPU。改成 `131072`
  + `--token-embedding-type q8_0`(让它是 HTP 能 repack 的类型)→ 上 NPU。**两法验证**(CPU 线程扩展
  16.9→35.6 tok/s + NPU profile 实测 1.43ms/次):同一 GEMV,**NPU 比 CPU 快 8.7×**。输出文本正确。
  → 这是可提上游的 patch(注释自己写着"for now")。**对 8-bit(Q8_0)同样有效**(设备上有 `Q8_0-embq8.gguf`)。
- **`GGML_HEXAGON_OPPOLL=1`(忙轮询同步):轻载 +18%**(消掉主机等 DSP 完成的中断唤醒延迟)。

**权重带宽的量化诊断** [实测/推理]:llama Q8_0 decode 有效带宽 ≈ `1.19GB × 26.6 ≈ 31.7 GB/s`,
只有 DDR 峰值(~80)的 **~40%**。**为什么只有 40% —— 这是接手方要 MEASURE 的核心问题(§7 #2)。**

---

## 4. 现有 tilelang↔Hexagon 对接设计(反思对象)

### 4a. 两层 DSL 抽象 [源码,已 device-validated]

- **Tier 1 · `T.gemm` → `GemmHMX`**(`tilelang/hexagon/gemm_hmx.py`):最简。用户写 GPU 风格
  `alloc_shared`(→VTCM)+ `T.copy` + `T.gemm(A,B,C, clear_accum=True)`。融合的 dequant 就是
  `T.gemm` 前的普通 DSL 算术。**故意 monolithic**:operands 保持 row-major(`infer_layout→{}`),
  Crouton pack/MAC/unpack 藏在 C 模板里。`GemmHMX` 只接受 fp16、32 倍数、2D、clear_accum。
- **Tier 2 · `HMXIntrinEmitter`**(`tilelang/hexagon/hmx_intrin.py`):instruction-atom,用户**手驱 K-loop**
  (`acquire / load_bias / clear / mma / convert / store / release`)。CUDA
  `TensorCoreIntrinEmitter` 的 Hexagon 对应。A/B/C 用 `T.Layout` 表达 Crouton 地址；accumulator、
  convert、bias 是 1-byte dependency token,只给 TIR 表达隐式硬件状态的顺序和存活期,不是可寻址 fragment。
  raw Q4_0 tile 在 DMA 落到 VTCM 后、`mma` 前由 kernel 中显式的 TileLang 循环完成 scale bitcast、
  low/high nibble 解包、减 8 和乘 scale；`T.Layout` 把逻辑 `(k,n)` 写入直接改写到最终 weight
  Crouton。这里没有 opaque dequant builtin、row-major FP16 B 或 `pack_b`。当前 correctness kernel 为每个 K tile 保留独立 B
  Crouton,直到 `store` 才释放;有 completion token 的 ping-pong/worker overlap 是下一步性能调度,不能
  通过复用一个仍被 HMX 消费的 buffer 来假装异步。
- 底层 runtime:两者都 lower 到 `hmx.h` 的 `tl_hexagon_hmx_*` 原语(**全 `__fp16`**)。
- HMX activation/weight/result Crouton 基址必须 2 KB 对齐,bias 必须 256 B 对齐;activation 与 result
  使用同一个 spatial-mask 编码。FP32 边界按 32 lane 转换,避免跨两个 HVX register 的 C vector ABI
  重排。
- **worker-pool over 6 HVX**(`T.Kernel(num_workers=N)`)+ per-worker VTCM:**device-validated**
  [见 memory `hexagon-backend-design`]。这是 decode 抽象要复用的关键件。

可运行例子:`example_qmatmul.py`(fused q4_0 matmul,Tier 1)、`example_qmatmul_kstream.py`(Tier 2)、
`example_worker_pool.py`、`example_flash_attention.py`、`example_matmul.py`、`example_rmsnorm.py`。

### 4b. llama.cpp 对接机制("Mode B",当前显式 Q4 correctness PoC)[源码/实测]

流程见 `docs/lfm2_hexagon_report/tilelang_operator_integration.md`(6 步完整 walkthrough)。要点:
1. DSL 算子 → `tilelang.compile(..., target="hexagon")` → `k.get_kernel_source()` **就是**可嵌入的
   `extern "C"` body(用 `tl_vtcm_base()` + `tl_hexagon_hmx_gemm`,无 skel/session 包装)。
2. **Mode B**:kernel 作为**源码** co-compile 进 host 的 `libggml-htp-v79.so`,**骑** host 已获取的
   VTCM 区 + HMX 锁(`tl_bridge_enter`,~6 行),不自己 acquire session。这是能嵌进别人 skel 的前提。
3. registry:`tl_op_desc {matches, run}` 自注册;`run()` 返回 `-1` = 声明不处理 → **回退 backend 原内核**
   (安全边界:tilelang 路径可只覆盖部分形状,不影响正确性)。
4. **拦截点** [关键]:patch 在 `matmul-ops.c` 的 **`hmx_mm_2d_f32(...)` 顶部**插 `if (tl_dispatch(&octx)==0) return 0;`。
   门控:`m>0 && m%32==0 && weight==Q4_0 && ...`;adapter 把完整 prefill `M` 循环切成固定
   `M=32` kernel 调用,所以真实的 `M=1024` 等形状也会进入 TileLang 路径。decode `M=1` 仍回退。

**现有对接的状态与局限(反思):**
- ✅ bridge(Mode B,骑资源)是真正可复用、可移植的核心。
- ⚠️ **只拦 prefill**(`hmx_mm_2d_f32`)。**decode(M=1 GEMV)走的是另一条 HVX 路径,当前完全没对接** —— 
  接手方要**找到 decode GEMV 的对接 seam**(见 §7 #4)。
- ⚠️ 是 **q4_0** 的,且当前只是 **correctness/integration PoC**。显式 dequant/HMX atom 版本在
  LFM2 `pp32` 为 2.85 t/s,stock 为 590.33 t/s,远未达到 parity。文中更早的 parity 结论属于
  monolithic `T.gemm` 历史路径,不能外推到当前低层实现。
- ⚠️ hmx.h 的 session state 是 `static`,每个嵌入 op 要在一个 TU 里(多 op 需 `tl_ctx*` 重构)。

---

## 5. HVX 铁律 + 踩过的坑(写 kernel 必读,否则设备上静默出错)

- **最窄 dtype 必须填满整个 128B 寄存器** [源码/实测]:uint8/int8 需 ≥128 lane,用 `T.vectorized(128)`;
  否则位运算 over-read 触发 fault。**Q8_0 的好处**:int8 是字节对齐的,dequant 只是 `(int8 * scale)`,
  比 q4_0 的 nibble 拆解简单得多 —— 对干净的 tilelang kernel 更友好。
- **bitwise 前先加宽到 int16**(q4_0 的坑;Q8_0 不拆 nibble,这条基本不涉及)。
- **VTCM tile 必须 `@T.macro`**:emitter 方法要 build+return 嵌套 `@T.macro`,否则 VTCM liveness pass
  看不到 scratch buffer 的使用 → 别名其它 `alloc_shared` → **静默垃圾**(不是崩溃)。
- **对齐**:HVX 整寄存器从 malloc 的 DDR 加载需 ≥128B 对齐(`memalign(256,…)`),否则静默垃圾。
- **`T.gemm` 必须 `clear_accum=True`**:HMX 累加器要先 `mxclracc`,否则 NaN。
- **测 compute 不是单次墙钟**:单次 `kernel()` 被 FastRPC input marshal 主导,曾两次误判性能。要 amortize
  或用常驻权重(即在模型里测)。**这条对 decode 尤其致命** —— 真实收益只在模型里、权重常驻时才显现。

---

## 6. 我的当前判断:8-bit decode 的正确抽象(起点,需接手方深化)

**核心论点:decode 的正确抽象不是矩阵引擎,是"逼近峰值带宽的权重流式流水线"。**

decode GEMV 是访存 bound(§2),llama 只跑到 40% 峰值(§3)。要逼近 nexa(~打满带宽),需要:

```
对每个 N-tile(切分以填满 VTCM + 跨 6 个 HVX 线程并行):
    DMA 预取下一个 W-tile 进 VTCM      ← 双缓冲,和下面的计算重叠(关键!)
    dequant 当前 W-tile(int8 → fp16,× per-block scale)   ← HVX,藏在 DMA 下
    accumulate  A · W-tile → y[N-tile]                      ← HVX FMA 归约
```

这套抽象需要的件:
1. **worker-pool over 6 HVX**(`T.Kernel(num_workers=N)`)—— **已有,device-validated**。
2. **per-worker VTCM** —— **已有**。
3. **DSL int8 dequant** —— 平凡(比 q4_0 简单)。
4. **DMA 异步 + 双缓冲(software pipeline)** —— **这是关键的、可能缺失的一块**(§7 #3):
   tilelang 在 CUDA 上有 `T.Pipelined`/async copy;**Hexagon 后端有没有对应的 DMA-double-buffer 抽象,
   必须查**。这决定了能不能把权重流式打满 —— 也就是能不能赢。
5. **HVX 归约(dot/FMA)** —— 需要一个高效的 M=1 点积 kernel。

**为什么不是 HMX**:M=1 喂 32×32 tile 浪费 97%;且访存 bound 时算力引擎不解决问题。HMX 留给 prefill。

**诚实的风险** [推理]:如果 llama 的 40% 带宽损失主要来自 **ggml 的权重布局 / 缺 DMA 预取**(而非
kernel 计算),那"赢"可能需要**改 ggml 的权重 staging**,而不只是写个 tilelang kernel。所以 §7 #2
(诊断带宽损失在哪)必须**先做**,它决定了 tilelang 到底能不能在这里赢、赢多少。

---

## 7. 必须先解决的 4 个开放问题(按优先级 + 怎么解)

**#1 [先做] llama.cpp 的 decode 到底在哪损失了 60% 带宽?** 这决定一切。
- 手段:(a) 用已有的 **AXI PMU 计数器**(`GGML_HEXAGON_PROFILE=2` 的 pmu 数组里,事件 `0x42`=AXI_write、
  `0x3f`=AXI_line128_read;见 `docs/hexagon_profiling.md` 的 PMU 解码)量 decode 的实际 DDR 读带宽 vs 峰值。
  (b) 写一个**纯 VTCM copy 带宽测试**(tilelang 或 ggml)看能打到多少峰值,作为上限参照。
  (c) 看 ggml-hexagon 的 decode GEMV 源码(`matmul-ops.c` 里 M=1 / vec_dot 路径),它有没有 DMA 预取/双缓冲。
- 结论若是"dequant 没藏好 / 没预取" → tilelang kernel 能赢;若是"权重布局烂" → 要改 ggml staging。

**#2 [先做] decode GEMV 的 ggml 对接 seam 在哪?** 当前只拦 `hmx_mm_2d_f32`(prefill)。
- 手段:读 `ggml/src/ggml-hexagon/htp/matmul-ops.c`,找 M=1 / q8_0 的 matmul 分派函数(可能是
  `hvx_mm_*` 或 vec_dot),那才是 decode GEMV 的拦截点。参考现有 patch 的做法(`ggml-hexagon.patch`)。

**#3 tilelang Hexagon 后端有没有 DMA 异步 / software-pipeline 抽象?**
- 手段:读 `tilelang/hexagon/pipeline.py`、`src/tl_templates/hexagon/`、grep `dma`/`async`/`Pipelined`/`hex-dma`。
  ggml 侧有 `htp/hex-dma.c`(可参考 DSP DMA 用法)。若 tilelang 没有,**这是要补的核心抽象**。

**#4 Hexagon HMX 是否支持 int8 矩阵模式?**(优先级低 —— 即便支持对 M=1 GEMV 也无用,但影响 prefill 8-bit)
- 手段:grep Hexagon SDK `incs/` 和 `hexagon_protos.h` 找 `mxmem`/`Q6_MX`/int8 变体;查 HMX v79 文档。
  我查到 `tl_templates/hexagon/hmx.h` **只有 fp16**;SDK 那边没查完(被打断)。

---

## 8. 建议的执行路径(想清楚 §7 之后)

1. **先摘 lm_head(Q8_0 版)** —— 已验证 +80% 且对 8-bit 有效。把「改上限 + `--token-embedding-type q8_0`」
   整理成干净 patch(可上游)。这不是 tilelang kernel,但是**最快、已证的 decode 收益**,且给后面的
   tilelang GEMV 一个「lm_head 上 NPU 后」的公平基线。设备上已有 `LFM2-1.2B-Q8_0-embq8.gguf`。
2. **诊断带宽(§7 #1)** —— 决定 tilelang 能否赢。
3. **第一个 tilelang 算子:一个 Q8_0 decode GEMV**(建议 FFN `down` 或一个投影,单个干净 GEMV,M=1)。
   目标:掌握 §6 的流式抽象,在模型里测对 backend HVX GEMV 的提升。**成了再谈子图融合**
   (短卷积块 `in_proj→conv→out_proj`,唯一没被 ggml 融的,15%)。
4. **对齐 nexa**:全程用 Q8_0,和 `24.1 → 69.5 tok/s @d1024` 这条线比。

---

## 9. 环境 · 复现 · 关键命令

**代码库**:
- tilelang(本仓库):`/home/xwh/tilelang-hexagon`,分支 `hexagon-backend`。
- llama.cpp(host 侧可改可重编):`/home/xwh/scratchpad-llamacpp`,commit `4fc4ec55`,Android 构建目录
  `build-snap/`(NDK r25c + Hexagon SDK 6.6)。**已改过** `ggml-hexagon.cpp:2797`(lm_head 上限 32768→131072)。
- cmake:`/home/xwh/.local/cmake/bin/cmake`(PATH 里没有,要全路径)。`source /tmp/hexenv.sh` 设 adb/SDK 环境。
- Hexagon SDK:`/home/xwh/Downloads/Hexagon_SDK_Linux/Hexagon_SDK/6.6.0.0`;DSP c++/工具在
  `.../tools/HEXAGON_Tools/19.0.07/Tools/`。

**设备**(adb;偶尔 USB 掉线,`adb kill-server; adb start-server; adb wait-for-device` 重连):
部署目录 `/data/local/tmp/llamahtp/`。已就绪:
- 模型:`LFM2-1.2B-{F16, Q4_0, Q8_0, Q4_0-embq4, Q8_0-embq8}.gguf`(embq* = lm_head 重量化版)。
- 二进制:`llama-cli`、`llama-bench`、`llama-quantize`(都是 Android build)。
- **lib 状态**:`libggml-hexagon.so` 当前 = `.new`(改过上限的);`.orig` 是原版备份。A/B 用 `cp` 切换。

**重编 host lib**(改 ggml-hexagon.cpp 后,不用动 skel):
```bash
source /tmp/hexenv.sh
/home/xwh/.local/cmake/bin/cmake --build /home/xwh/scratchpad-llamacpp/build-snap --target ggml-hexagon -j$(nproc)
adb push /home/xwh/scratchpad-llamacpp/build-snap/bin/libggml-hexagon.so /data/local/tmp/llamahtp/
```
**重编 skel**(改 tilelang kernel 对接进 skel 时):`--target htp-v79`,推 `libggml-htp-v79.so`。

**性能测试**(decode @1024 上下文,对齐 nexa):
```bash
adb shell "cd /data/local/tmp/llamahtp && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp \
  ./llama-bench -m LFM2-1.2B-Q8_0.gguf -dev HTP0 -ngl 99 -n 128 -d 1024 -r 3"
# 加 GGML_HEXAGON_OPPOLL=1 拿轻载 +18%
```

**逐算子 PMU 剖析**(踩坑:必须 `GGML_HEXAGON_VERBOSE=1` **和** `--verbose`;log 是 non-ISO ASCII 要 `grep -a`;
**融合算子名含 `+`,正则要 `[\w+]+` 不是 `\w+`**,否则漏 40% 算子):
```bash
adb shell "cd /data/local/tmp/llamahtp && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp \
  GGML_HEXAGON_PROFILE=2 GGML_HEXAGON_VERBOSE=1 \
  ./llama-cli -m LFM2-1.2B-Q8_0.gguf --device HTP0 -ngl 99 -n 16 -st --verbose -f prompt_1024.txt 2>&1" | tr -d '\r' > prof.log
```
现成脚本:`docs/lfm2_hexagon_report/scripts/`(`run_bench.sh`/`run_profile.sh`/`run_op_split.sh`/
`make_q8.sh` + 解析器 `profile_lfm2.py`/`timeline_export.py`/`chrome_trace_export.py`/`forward_pass.py`)。

**关键调度旋钮** [源码 `ggml-hexagon.cpp`]:`GGML_HEXAGON_OPPOLL`(0/1 忙轮询)、`GGML_HEXAGON_OPBATCH`(1024)、
`GGML_HEXAGON_OPQUEUE`(16)、`GGML_HEXAGON_OPSTAGE`(1=只排队不算,用于分离框架开销)、
`GGML_HEXAGON_OPFUSION`(1)、`GGML_HEXAGON_PROFILE`(1/2)、`GGML_HEXAGON_VERBOSE`。

---

## 10. 文件地图

**tilelang 后端**:`tilelang/hexagon/{hmx_intrin.py, gemm_hmx.py, pipeline.py, adapter.py, mini_htp.py, ...}`;
runtime 模板 `src/tl_templates/hexagon/{hmx.h, tl_bridge.h, tl_embed.h, common.h}`;codegen
`src/hexagon/`、`src/transform/loop_vectorize.cc`。

**例子/对接**:`examples/hexagon/*.py`(6 个);`examples/hexagon/llama_cpp_integration/`(对接全套:
`emit_embeddable.py`、`tl_ggml_matmul.cc`、`ggml-hexagon.patch`、`kernel_*.c`、`.manifest.json`)。

**本 session 产物**(`docs/`):
- `lfm2_hexagon_report/` —— 中文报告 README + `scripts/` + `results/`(原始 log)+ **汇报 slides**
  `tilelang_npu_slides.html` + **算子对接 walkthrough** `tilelang_operator_integration.md`。
- `hexagon_lfm2_perf.md`(英文完整报告)、`hexagon_profiling.md`(方法学 + PMU 解码 + itrace 尝试)、
  `hexagon_timeline.html`(交互 nsys 时间线)、`lfm2_perfetto_trace.json`(Perfetto)。
- `hexagon_backend_summary.md` / `hexagon_dsl_kernels.md` / `llama_cpp_integration.md` —— 之前的设计文档。

**背景设计文档(之前的)**:`docs/hexagon_backend.md`、`edge_tilelang_vs_cuda.md`、`lfm2_1.2b_compute_graph.md`。

**Memory**(`/home/xwh/.claude/projects/-home-xwh-tilelang-hexagon/memory/`):`MEMORY.md` 索引 + 8 个
memory(hexagon-backend-design / hmx-programming-model / hmx-tile-ops / hvx-dequant-primitive /
fused-q4_0-matmul / llama-cpp-hexagon-llm-runtime / **lfm2-hexagon-perf-profile**(本 session 的核心发现)等)。

**本 session 提交**:`5c799fed` → `46ce4b49`(见 `git log`),都是 `docs/` 的剖析报告 + slides + 交接。
llama.cpp 那边改的 `ggml-hexagon.cpp:2797` 上限**还没提交/整理成 patch**(在 `/home/xwh/scratchpad-llamacpp`)。

---

## 11. 诚实的边界(接手方要知道什么不可信)

- **没在设备上实跑过 nexa**(license 校验失败),所有 nexa 数字是官方报告值,内部机理("用了 HMX"/"整图编译")
  是 **[存疑]** 推测。差距的**大小**是实测的,**归因**是推理的。
- **DDR 峰值 ~77–85 GB/s 是参考值**,SM8750 的确切值接手方应查证(影响"40% 峰值"这个诊断)。
- **decode "40% 带宽"是从 tok/s 推的有效带宽**,不是直接测的 AXI 带宽 —— §7 #1 要用 PMU 直接测。
- **单算子 parity 是天花板**这条对 q4_0 成立(backend 已调优);**Q8_0 的 decode GEMV backend 是否也已调优
  到 nexa 水平,未知** —— 从"llama Q8_0 只有 nexa 的 40%"看,大概率没有,这正是机会,但需 §7 #1 证实。
