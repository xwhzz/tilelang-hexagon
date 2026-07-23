# llama.cpp 在 Hexagon 上的完整推理流程

从「调用模型」到「吐出一个 token」的全链路,配合 LFM2-1.2B 的真实计算图与实测的算子分布
(`GGML_SCHED_DEBUG=2` 抓的,不是猜)。术语:host = ARM CPU 侧;HTP/cDSP = Hexagon NPU;
skel = 跑在 cDSP 上的 FastRPC 库 `libggml-htp-vNN.so`。相关源码:`ggml/src/ggml-hexagon/`。

配套:算子分布实测见 `examples/hexagon/llama_cpp_integration/lfm2_1.2b_op_split.md`;
把 tilelang 算子接进来的机制见 `docs/llama_cpp_integration.md`。

---

## 0. 全景:五层

```
 llama-cli (host)
   │  1. 读 GGUF(权重 + 元数据,权重已离线量化)
   │  2. 逐层把权重分配到某个 backend 的 buffer(-ngl / --device 决定)
   │     └─ 放进 HTP 的 REPACK buffer 时,host 把权重 repack 成 HMX tile 版式(见 §2)
   │
   ├─ llama_decode(tokens)               每步推理入口
   │     └─ 3. 构建 ggml 计算图(LFM2 结构)
   │
   ├─ ggml_backend_sched(ggml 核心,不是 hexagon 后端)
   │     └─ 4. 逐节点选 backend:supports_op + 数据所在 buffer → 打 [CPU]/[HTP0] 标
   │         切成 "split"(同 backend 的连续段),边界自动插 CPU↔HTP 拷贝
   │
   ├─ ggml-hexagon 后端(host 侧)
   │     └─ 5. 对分给 HTP 的 split:融合 → 映射 GGML_OP→HTP_OP → 压进 FastRPC 命令队列
   │
   └─ FastRPC(异步 dspqueue + rpcmem 共享内存,零拷贝)
         │
         └─ cDSP skel:6. 取队列 → switch(op) → HVX/HMX kernel → 写回共享 buffer
```

三条关键真相:**① 切 CPU/HTP 的是 ggml 核心的 scheduler,不是 hexagon 后端**(后端只提供
`supports_op` + buffer 类型);**② 传输是异步命令队列 + 共享内存,不是一 op 一次阻塞 RPC**;
**③ 权重按 op 传的是"共享 buffer 里的指针",不逐 op 拷贝**,只有 CPU↔HTP 边界才拷。

---

## 1. 量化(quantization):权重为什么要变小

**动机:LLM decode 是"内存带宽瓶颈"** —— 每吐一个 token 要把**整套权重从 DDR 流一遍**过计算
单元。算得多快无所谓,喂权重的带宽才是瓶颈。所以核心是**每个权重占几个字节**:字节越少,decode 越快。

- **q4_0**(LFM2 的主力):一个 block = 32 个权重,存 `{ fp16 d; uint8 qs[16] }` = **18 字节**
  (2 字节尺度 + 16 字节 = 32 个 4-bit nibble),即 **~4.5 bit/权重**。反量化:
  `weight[i] = d * (nibble[i] - 8)`(nibble 0..15,减 8 对称到 -8..7)。
- **q6_K**(LFM2 的 embedding 和 LM head):6-bit 的 K-quant,superblock 结构(256 权重一块,
  含子块尺度),**~6.5 bit/权重**,精度更高 —— embedding/输出头对量化敏感,llama.cpp 故意留高精度。
- **GGUF 里存的就是量化后的字节**(离线量化器产出),llama.cpp 加载时原样读入,推理时按需反量化。

**量化在后端里的作用**:`ggml_backend_hexagon_device_supports_op` 会看权重 dtype。
`q4_0/q4_1/q8_0/iq4_nl/mxfp4` 是"可 repack 类型"(`ggml_hexagon_is_repack_type`,`:163`)→ HTP 能做;
**`q6_K` 不是** → 后端没有 q6_K 的 HMX kernel → `supports_op` 返回 no → 落 CPU。
这正是实测里唯一两个落 CPU 的节点(embedding + LM head,都是 q6_K)的根因。

---

## 2. repack:为什么量化完还要再排一次版

存在 GGUF 里的 q4_0 是 llama.cpp 的**标准 block 版式**(每 18 字节一个 block,尺度和 nibble 交错)。
但 **HMX 矩阵引擎要的是 tile 版式**(32×32 tile,nibble 和尺度分开、按列打包),直接喂标准版式没法高效 MAC。

所以**加载时**,当一个 q4_0 权重被放进 HTP 的 REPACK buffer,host 的 `repack_q4_0_tiled`
(`ggml-hexagon.cpp:446`)把它重排成 **576 字节/tile**:

```
一个 32(输出特征) × 32(K) 的 tile:
  nibble 区(512B): byte[cp*32 + row] = (quant[row][2cp+1] << 4) | quant[row][2cp]   cp=0..15,row=0..31
  尺度区(+512):    32 个 fp16,scale[row]                                            (每个输出特征一个)
整块权重按 (col_tile, k_tile) 排成 (ct*n_k_tiles+kt)*576 的连续 tile 数组
```

要点:**repack 只在加载时做一次**(不是每 token);做完权重就住在 DDR 的 rpcmem 里,DSP 直接读。
这也解释了 `supports_op` 为什么对同一个 q4_0 matmul 会打两次:非 repack buffer 变体 → no,
`HTP0-REPACK` buffer 变体 → yes —— 权重必须在 repack buffer 里才走 HTP。

---

## 3. 加载模型(一次性)

1. **解析 GGUF** → 每个张量的形状/dtype + 量化字节。
2. **分配 backend buffer**:`-ngl N` / `--device HTP0` 决定哪些层的权重放到 HTP session 的 buffer。
   对可 repack 的 quant 类型,用的是 **REPACK buffer type**(`ggml-hexagon.cpp:221`)。
   8B 这种大模型用 `GGML_HEXAGON_NDEV=4` 把层摊到 HTP0..3 四个 session(绕 32-bit cDSP 的 4GB 地址空间)。
3. **写权重 → 触发 repack**:REPACK buffer 的 `set_tensor` 调 `repack_*_tiled`,把权重排成 tile 版式,
   落在 **rpcmem/ION**(host 分配、映射进 DSP 保护域,零拷贝共享)。
4. **起 DSP skel**:FastRPC 加载 `libggml-htp-vNN.so`,每 session 申请 **VTCM(8MB 片上)+ HMX + 电源**
   (`main.c::vtcm_alloc`,一把 `HAP_compute_res` 同时拿 VTCM 和 HMX)。

---

## 4. 一步推理:`llama_decode` → 吐一个 token

```
llama_decode(token)
  └─ 构图: 按 LFM2 结构建这一步的 ggml 计算图(embed → 16 层 → final norm → LM head)
      │
      └─ ggml_backend_sched_graph_compute(graph)
          ├─ 逐节点定 backend: supports_op(op) && 三个 src/dst 都在同一 HTP session 的 buffer
          │                     → [HTP0];否则 → [CPU];边界插拷贝
          ├─ 切 split(同 backend 连续段)
          └─ 对每个 HTP split → ggml_backend_hexagon_graph_compute (:3551)
               ├─ try_fuse_node (:3473):RMS_NORM+MUL→RMS_NORM_MUL、MUL_MAT+ADD→MUL_MAT_ADD、
               │                         QKV 三矩阵→MUL_MAT_QKV、FFN→MUL_MAT_FFN
               ├─ op_remap_to_htp (:3346):GGML_OP_* → HTP_OP_*
               ├─ 预计算 kernel 参数(matmul 的 tile solver / flash-attn 参数),按 graph uid 缓存
               └─ enqueue_op → dspqueue_write (:1577)   ← 压进异步 FastRPC 队列
      │
   [cDSP skel]  htp_packet_callback 取队列
      └─ switch (octx->op) (main.c:576):
           HTP_OP_MUL_MAT/MUL_MAT_ADD → op_matmul   (dequant q4_0→fp16 + HMX MAC + f32 out)
           HTP_OP_MUL_MAT_ID          → op_matmul_id (MoE 专家)
           HTP_OP_RMS_NORM_MUL/…      → op_unary / op_binary / op_activations(HVX)
           HTP_OP_FLASH_ATTN_EXT      → op_flash_attn_ext(HMX+HVX)
           HTP_OP_SSM_CONV            → op_ssm_conv(HVX)
           → kernel 从 VTCM/DDR 读、算、把结果写回共享 buffer,回 op_status
      │
  host: CPU split(本模型就是 q6_K 的 embedding + LM head)照常在 CPU 跑
      └─ 得到 logits → 采样 → 下一个 token
```

**实测(LFM2-1.2B,一张 prefill 图 354 个节点):HTP0=352,CPU=2。** 整个 transformer + short-conv +
attention 栈全在 NPU;只有 q6_K 的 `GET_ROWS model.embed_tokens` 和 `MUL_MAT result_output`(LM head)在 CPU。

---

## 5. 计算图(LFM2-1.2B 真实结构,含设备标注)

16 层,交替 **short-conv 块**(10 个)和 **GQA attention 块**(6 个,实测 flash-attn 落在 6 层);
每个块后面都接一个 **SwiGLU FFN**。`[H]`=HTP0,`[C]`=CPU。

```
输入 token
  └─ GET_ROWS  embed_tokens                         [C]   ← q6_K,唯一在 CPU 的查表
      │
      ├──────────── short-conv 块(×10) ────────────
      │   RMS_NORM → MUL(norm)                        [H]
      │   MUL_MAT  in_proj  (2048→3*2048, q4_0)       [H]  ← 拆成 B,C,x
      │   MUL(门控 B·x) → CONCAT(接卷积状态)→ CPY(写 cache_r) [H]
      │   SSM_CONV (短因果卷积)                        [H]
      │   MUL(门控 C) → MUL_MAT out_proj (2048→2048)   [H]
      │   ADD(残差)                                    [H]
      │   └─▶ [FFN]
      │
      ├──────────── attention 块(×6) ───────────────
      │   RMS_NORM → MUL(norm)                        [H]
      │   MUL_MAT  Qcur/Kcur/Vcur (q4_0)              [H]  ← 后端会融成 MUL_MAT_QKV
      │   ROPE                                         [H]
      │   SET_ROWS cache_k / cache_v(写 KV cache)      [H]
      │   FLASH_ATTN_EXT                               [H]  ← HMX(score gemm)+HVX(softmax)
      │   MUL_MAT  out_proj → ADD(残差)                [H]
      │   └─▶ [FFN]
      │
      │   [FFN](每个块都有):
      │     RMS_NORM → MUL(ffn_norm)                   [H]
      │     MUL_MAT ffn_gate  +  MUL_MAT ffn_up (q4_0) [H]
      │     SWIGLU (silu(gate)·up)                     [H]
      │     MUL_MAT ffn_down → ADD(残差)               [H]
      │
      └─ (16 层之后)
  RMS_NORM(final) → MUL(result_norm)                  [H]
  MUL_MAT  result_output  (LM head, 2048→vocab, q6_K) [C]   ← 唯一在 CPU 的计算
  → logits → 采样
```

对照真实节点序(节选自 dump):
- conv 层:`#11 RMS_NORM → #12 MUL → #14 MUL_MAT(in_proj) → #17 MUL → #19 CONCAT → #22 CPY →
  #32 SSM_CONV → #33 MUL → #34 MUL_MAT(out_proj) → #36 ADD → #37 RMS_NORM → #39 MUL_MAT(ffn_gate)
  → #40 MUL_MAT(ffn_up) → SWIGLU → MUL_MAT(ffn_down) → ADD`
- attn 层:`RMS_NORM → MUL → #78 MUL_MAT(Kcur) → #82 ROPE → #84/#86 SET_ROWS(kv) → #93 FLASH_ATTN
  → MUL_MAT(out) → ADD → [FFN]`
- 收尾:`#544 ADD(l_out-15) → #545 RMS_NORM → #546 MUL(result_norm) → #547 MUL_MAT(result_output)[C]`

---

## 6. 数据怎么过去/回来(传输层)

- **共享内存 rpcmem/ION**:host 分配、映射进 DSP 的保护域,**零拷贝**。repack 后的权重、激活、KV cache、
  卷积状态都住这里;DSP 直接按物理地址读写。
- **FastRPC dspqueue(异步命令队列)**:host 把一个个 op 请求(共享 buffer 里的指针 + 形状/参数)压进队列
  (`dspqueue_write`),DSP 端 `htp_packet_callback` 连续取、执行、回 `op_status`。**不是一 op 一次同步 RPC**
  —— 整段 HTP split 压进去,DSP 流式消费,省掉每 op 的往返延迟。
- **只有 CPU↔HTP 边界才拷贝**:sched 在 [C] 和 [H] 交界处插 copy。本模型的边界很少(就 embedding→第一层、
  最后一层→LM head 附近),所以拷贝开销可忽略。

---

## 7. 一句话总结每个概念

- **量化**:把权重压到 ~4.5 bit(q4_0),因为 decode 卡在权重带宽 —— 少搬字节 = 快。
- **repack**:加载时把标准 q4_0 版式重排成 HMX 要的 32×32 tile 版式,一次性,之后住在共享内存。
- **调用模型**:`llama_decode` 建图 → ggml scheduler 逐 op 分 CPU/HTP → HTP 段异步压进 FastRPC 队列 →
  cDSP 派发到 HVX/HMX kernel → 写回共享内存 → CPU 收尾 → 采样。
- **一步推理**:embed(CPU)→ 16 层(全 NPU:conv/attention + SwiGLU FFN)→ final norm(NPU)→ LM head(CPU)→ logits。
- **哪些不在 NPU**:只有 q6_K 的 embedding 查表和 LM head 两个节点(后端没有 q6_K kernel);其余 352/354 全在 NPU。
