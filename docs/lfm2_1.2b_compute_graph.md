# LFM2-1.2B-Q4_0 计算图(形状 / 类型 / 节点 / 设备)

从设备实测抓的(`GGML_SCHED_DEBUG=2 GGML_HEXAGON_VERBOSE=1 -v`,Hexagon v79)。形状记法 `ne0:ne1`
(ggml 是列优先,`ne0`=最内维)。下面的激活形状是 **decode(M=1,单 token)** 的;prefill 时把每个
`… x 2048:1 -> …:1` 里的 `:1` 换成 `:seq_len`。设备标注 `[H]`=HTP0(NPU),`[C]`=CPU。

## 模型配置(从形状反推)

| 项 | 值 | 依据 |
|---|---|---|
| hidden `n_embd` | **2048** | 所有 norm/proj 的 2048 维 |
| FFN `n_ff` | **8192** | ffn_gate/up `2048→8192`,ffn_down `8192→2048` |
| vocab | **65536** | `token_embd 2048:65536` |
| 注意力 | **GQA**,Q=2048(32 头×64),KV=512(8 头×64),head_dim=64 | attn_q `2048→2048`,attn_k/v `2048→512`,attn_q/k_norm `64:1` |
| short-conv | in_proj `2048→6144`(=3×2048,拆 B/C/x),kernel=3,out_proj `2048→2048` | `shortconv.in_proj 2048:6144`,`conv.weight 3:2048` |
| 层数 | **16**,其中 attention 6 层{2,5,8,10,12,14}、short-conv 10 层 | flash-attn 落在 6 层 |
| 权重量化 | 矩阵权重 **q4_0**;norm/conv **f32**;embedding+LM head **q6_K**(共享/tied) | 见下表 |

## 每层的权重张量(形状 · 类型 · 设备)

一个 **short-conv 层**(如 blk.0):
| 张量 | 形状 | 类型 | 设备 |
|---|---|---|---|
| `attn_norm.weight`(层前 RMSNorm) | `2048:1` | f32 | H |
| `shortconv.in_proj.weight` | `2048:6144` | q4_0 | H |
| `shortconv.conv.weight` | `3:2048` | f32 | H |
| `shortconv.out_proj.weight` | `2048:2048` | q4_0 | H |
| `ffn_norm.weight` | `2048:1` | f32 | H |
| `ffn_gate.weight` | `2048:8192` | q4_0 | H |
| `ffn_up.weight` | `2048:8192` | q4_0 | H |
| `ffn_down.weight` | `8192:2048` | q4_0 | H |

一个 **attention 层**(如 blk.2)把上面的 short-conv 部分换成:
| 张量 | 形状 | 类型 | 设备 |
|---|---|---|---|
| `attn_norm.weight` | `2048:1` | f32 | H |
| `attn_q.weight` | `2048:2048` | q4_0 | H |
| `attn_k.weight` / `attn_v.weight` | `2048:512` | q4_0 | H |
| `attn_q_norm.weight` / `attn_k_norm.weight` | `64:1` | f32 | H |
| `attn_output.weight` | `2048:2048` | q4_0 | H |
| （FFN 同上：gate/up `2048:8192`,down `8192:2048`,q4_0） | | | |

**收尾 / 两头**:
| 张量 | 形状 | 类型 | 设备 |
|---|---|---|---|
| `token_embd.weight`(embedding 查表 + LM head,tied) | `2048:65536` | **q6_K** | **C** |
| `token_embd_norm.weight` / `output_norm` | `2048:1` | f32 | H |

## 节点级计算图(带算子的输入/输出形状,decode M=1)

`OP  src0(名,形状,类型) × src1 → dst(形状,类型)  [设备]`

### short-conv 块(blk.0)
```
RMS_NORM   x(2048:1,f32)                                   -> norm(2048:1,f32)            [H]
MUL        norm × attn_norm.weight(2048:1,f32)             -> (2048:1,f32)                [H]
MUL_MAT    in_proj.weight(2048:6144,q4_0) × (2048:1,f32)   -> bcx(6144:1,f32)             [H]   # 拆成 B,C,x 各 2048
MUL        B × x                                           -> Bx(2048:...,f32)            [H]   # 门控
CONCAT     conv_state × Bx                                 -> (…,f32)                     [H]   # 接卷积状态
CPY        -> cache_r_l0(4096:1,f32)                       (写回卷积状态缓存)              [H]
SSM_CONV   conv.weight(3:2048,f32) × (…)                   -> (2048:1,f32)                [H]   # 短因果卷积,kernel=3
MUL        C × conv_out                                    -> (2048:1,f32)                [H]   # 门控
MUL_MAT    out_proj.weight(2048:2048,q4_0) × (2048:1,f32)  -> (2048:1,f32)                [H]
ADD        + 残差                                          -> (2048:1,f32)                [H]
── FFN ──
RMS_NORM   -> (2048:1)   MUL ×ffn_norm.weight(2048:1)                                     [H]
MUL_MAT    ffn_gate.weight(2048:8192,q4_0) × (2048:1,f32)  -> gate(8192:1,f32)            [H]
MUL_MAT    ffn_up.weight  (2048:8192,q4_0) × (2048:1,f32)  -> up(8192:1,f32)              [H]
SWIGLU     silu(gate)·up                                   -> (8192:1,f32)                [H]
MUL_MAT    ffn_down.weight(8192:2048,q4_0) × (8192:1,f32)  -> (2048:1,f32)                [H]
ADD        + 残差                                          -> (2048:1,f32)                [H]
```

### attention 块(blk.2)
```
RMS_NORM   x -> norm(2048:1,f32)   MUL ×attn_norm.weight                                  [H]
MUL_MAT    attn_q.weight(2048:2048,q4_0) × (2048:1,f32)    -> Q(2048:1,f32)  (32头×64)    [H]
MUL_MAT    attn_k.weight(2048:512, q4_0) × (2048:1,f32)    -> K(512:1,f32)   (8头×64,GQA) [H]
MUL_MAT    attn_v.weight(2048:512, q4_0) × (2048:1,f32)    -> V(512:1,f32)                [H]
RMS_NORM   Q,K 各头 ×attn_q_norm/attn_k_norm.weight(64:1,f32)                             [H]
ROPE       Q,K                                             -> 同形                        [H]
SET_ROWS   K,V -> cache_k_l2 / cache_v_l2                  (写 KV cache)                  [H]
FLASH_ATTN_EXT  Q × K^cache × V^cache                      -> (2048:1,f32)   HMX+HVX      [H]
MUL_MAT    attn_output.weight(2048:2048,q4_0) × (2048:1)   -> (2048:1,f32)                [H]
ADD        + 残差                                                                        [H]
── FFN ──（同上）
```

### 两头(唯一在 CPU 的计算)
```
GET_ROWS   token_embd.weight(2048:65536,q6_K) × tokens(i32) -> (2048:1,f32)   embedding   [C]
   … 16 层 …
RMS_NORM   -> (2048:1)   MUL ×output_norm                                                 [H]
MUL_MAT    token_embd.weight(2048:65536,q6_K) × (2048:1,f32) -> logits(65536:1,f32)  LM头 [C]
```

## 一句话读法

- **算力全在 q4_0 的 MUL_MAT 上**:每层 ~7 个矩阵乘(conv 层:in_proj `2048×6144` + out_proj `2048×2048` + FFN 三个 `2048×8192`/`8192×2048`;attn 层:QKV + O + FFN)。这些 decode 时是 **M=1 的 GEMV(访存瓶颈,权重是主要字节流)**。
- **只有两头 q6_K 落 CPU**:embedding 查表 + LM head(`2048×65536`,词表大所以这个矩阵最大,105M 权重),后端无 q6_K kernel。
- **其余全在 NPU**:norm/门控/卷积/softmax/rope 都是 f32 的 HVX 小算子,矩阵乘走 q4_0→HMX。

配套:算子↔设备分布见 `lfm2_1.2b_op_split.md`(352/354 在 NPU);整条推理链见 `docs/llama_cpp_inference_flow.md`。
