# TileLang Hexagon Q8 decode：后端边界、硬件映射与模型集成

日期：2026-07-15。设备：OnePlus PJZ110 / SM8750 / Hexagon v79。

本文是对 [`tilelang_hexagon_8bit_handoff.md`](tilelang_hexagon_8bit_handoff.md) 的复核结果，
也是当前 Q8_0 实现的设计说明。结论不是“重写整个 Hexagon backend”，而是先把语义、布局、
指令 atom、流水线策略和宿主资源所有权分开，再逐层替换并测量。

## 1. 先纠正基线

### 1.1 旧 Q8 decode 混入了 CPU `lm_head`

ggml-hexagon host backend 原来有硬编码：

```cpp
if (src0->ne[1] > 32768) return false;
```

LFM2 的词表为 65536，因此输出头被赶回 CPU。把实验上限提高到 131072 后，在同一设备、同一
模型、同一 `llama-bench -p 1024 -n 128 -r 3` 口径下：

| 配置 | pp1024 | tg128 |
|---|---:|---:|
| 原 host 库（32768 限制，单次复核） | 3290.28 | 28.17 |
| 新 host 库（允许 65536 `lm_head`） | 3288.12 ± 124.96 | **35.68 ± 0.01** |
| 新 host 库 + `Q8_0-embq8` | 3279.22 ± 113.88 | 35.37 ± 0.15 |

所以 `embq8` 后缀不是主要变量；是否允许输出头进入 HTP 才是。旧文档中的 24–27 tok/s 不能
继续作为“纯 Hexagon Q8 GEMV”基线。

### 1.2 stock Q8 已经有完整流式流水线

`ggml/src/ggml-hexagon/htp/matmul-ops.c` 的 M=1 Q8 分派为
`hvx_mv_2d_repacked_q8_0`。其实际流程是：

1. 六个 worker 按 32-output-row tile 切分 N；
2. 每个 worker 用独立 DMA queue，把 DDR 中的 1088-byte Q8 tile row 以 2D DMA 搬到 VTCM；
3. DMA 同时把每个 K tile padding 到 1152 bytes；
4. `n_prefetch` 为 2–16 的 2 次幂，先填队列，再 `pop current / compute / push next`；
5. activation 在 VTCM 中动态量化为相同的 tiled Q8 格式；
6. `tiled_vec_dot_q8_0_32x1` 用 HVX `vrmpyacc` 做 signed-int8 dot，再乘 weight/activation scale；
7. 最后把 32 个 fp32 输出写回。

因此“给 `T.copy` 加一次 double buffer 就能补齐 2× 差距”不是成立的诊断。TileLang 若直接复制
这一整套代码，首先得到的只会是另一个 stock 实现。

### 1.3 不能用 `模型文件大小 × tok/s` 代替 DDR 测量

该估算会混入 CPU fallback、embedding/output weight tying、融合算子、cache traffic、padding、
量化 activation 和运行时空洞。当前默认 PMU 事件集合
`{0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C}` 也不包含旧文档建议的
`AXI_line128_read`，不能从已有 Q4 profile 反推出 Q8 的真实 DDR GB/s。要下带宽结论，必须先换
PMU event set，并用独立 DMA/copy microbenchmark 校准。

同样，nexa 的 69.5 tok/s 在缺少完全相同模型 artifact、量化细节、融合图和测量命令时只能作为
产品目标，不能宣称是严格同口径内核对比。

## 2. 硬件事实与软件含义

本地 SDK 文档：

- `$HEXAGON_SDK_ROOT/docs/pdf/80-N2040-60_REV_AA_Hexagon_V79_Programmer_Reference_Manual.pdf`
- `$HEXAGON_SDK_ROOT/docs/pdf/80-N2040-61_REV_AB_Hexagon_V79_HVX_Programmer_Reference_Manual.pdf`
- `$HEXAGON_SDK_ROOT/docs/pdf/80-N2040-62_AA_Hexagon_V81_HMX_Programmer_Reference_Manual.pdf`

SDK 根目录为 `/home/xwh/Downloads/Hexagon_SDK_Linux/Hexagon_SDK/6.6.0.0`。设备 runtime 报告
6 hardware threads、6 HVX contexts、1 HMX、8 MiB VTCM。

| 硬件 | 已确认事实 | 对 TileLang 的约束 |
|---|---|---|
| HVX | 1024-bit / 128-byte vector；int8 128 lane、fp16 64 lane、fp32 32 lane | dtype 不同，合法/高效的 vector width 不同；Q8 dot 应成为指令 atom，不能依赖普通 scalar reduction 自动变成 `vrmpyacc` |
| VTCM | 软件管理 scratchpad，无自动 cache 语义 | `alloc_shared` 可以表示存储层级，但搬运、生命周期和 per-worker slice 必须由 lowering/runtime 明确管理 |
| DMA | `dmstart/dmlink/dmpoll/dmwait`，descriptor 可做 2D stride/padding | 用户 DSL 应表达 async copy 与 pipeline 依赖，不应直接暴露 descriptor 位域 |
| HMX | 当前公开手册与 TileLang runtime 验证的是 fp16 32x32 Crouton 路径 | 适合有 M/N 重用的 prefill；不能把 M=1 Q8 decode 默认映射成 HMX GEMM |
| worker pool | 当前设备可并行六个 HVX context | `T.Kernel(num_workers=6)` 是执行映射，不等同于六个 GPU block/六份独立 HMX |

HMX 硬件是否还有未公开/未验证的 int8 模式不影响当前 decode 决策：M=1 且权重只使用一次时，
先解决的是数据流和 HVX dot，而不是强行复用 fp16 HMX abstraction。

## 3. 正确的 TileLang 分层

### L0：语义 op

长期 API 应表达类似：

```text
y[N] = dequant_q8_0(W[N,K]) @ quantize_q8_0(x[K]) + optional_bias[N]
```

语义包含量化格式、scale block、accum dtype 和 optional epilogue；不包含 `dmlink`、descriptor 地址或
具体 prefetch 深度。

### L1：目标布局

本次新增 `Q8_0TiledLayout`：

- logical tile：`N=32, K=32`；
- DDR/source tile：1024 quant bytes + 64 scale bytes = 1088 bytes；
- VTCM/staged tile：DMA padding 后 1152 bytes = 9 个 HVX vectors；
- activation tile：8 个 replicated int8 vectors + 1 个 replicated fp16-scale vector。

布局是 ABI，不是零散 pointer arithmetic。它集中验证 K 对齐并计算 staged buffer 大小。

### L2：目标指令 atom

`Q8GemvIntrinEmitter.dot_32x1` 对应 CUDA backend 的 MMA atom，而不是完整 GEMV：

```python
emitter = Q8GemvIntrinEmitter(K)
emitter.dot_32x1(dst, staged_weight, staged_activation, bias)
```

它 lower 到 `tl_hexagon_q8_0_dot_32x1`。runtime 使用：

- `Q6_W_vshuff_VVR` 重排 weight bytes；
- `Q6_Vw_vrmpyacc_VwVbVb` 并行完成 32 个 signed-int8 dot；
- fp16 weight scale × activation scale 转 fp32；
- K-block 间 fp32 accumulate；
- optional fp32 bias/residual。

这和现有 `HMXIntrinEmitter` 的层级一致：TileLang 负责调度组合，目标 template 负责不可合理拆成普通
算术循环的硬件指令序列。

### L3：流水线 lowering

完整 TileLang QGEMV 未来应由 `T.Pipelined` + global→shared `T.copy`/`T.async_copy` 表达：

```text
prefetch staged_weight[next] -> compute dot[current] -> recycle buffer
```

但当前不能仅把 `TargetHasAsyncCopy(hexagon)` 改成 true：

- `AsyncCommitWaitAttrLowerer` 仍硬编码 `ptx_commit_group/ptx_wait_group`；
- Hexagon `Copy::Lower` 仍是同步 `LowerNormalCopy`；
- DMA queue/descriptor 的 per-worker 生命周期还没有 backend runtime owner。

正确改法是先把 pipeline 的 commit/wait lowering 抽成 target hook，再让 Hexagon copy lowering选择 1D/2D
DMA，并由 kernel/session 持有 queue。raw `dmstart/dmlink` 不进入用户 DSL。

### L4：宿主资源租约

独立 TileLang FastRPC kernel 可以自己建立 session；嵌入 llama.cpp 时则必须“ride”宿主已经拥有的
VTCM、DMA queue 和 worker pool。谁申请、谁释放，不能跨边界混用。本次模型例子特意把 seam 放在
`dma_queue_pop()` 之后，因此 TileLang 不重复申请资源，也不改变 stock pipeline。

## 4. 已实现文件

| 文件 | 作用 |
|---|---|
| `tilelang/hexagon/qgemv.py` | `Q8_0TiledLayout`、`Q8GemvIntrinEmitter`、embeddable PrimFunc factory |
| `src/tl_templates/hexagon/qgemv.h` | v79 HVX Q8_0 `32x1` dot atom |
| `tilelang/hexagon/_fastrpc.py` | standalone kernel 自动包含 `qgemv.h` |
| `examples/hexagon/example_qgemv_q8_0.py` | Q8 pack、golden 和真实设备数值验证 |
| `examples/hexagon/llama_cpp_integration/emit_qgemv_q8_0.py` | 生成 K=2048/8192 embeddable kernels + manifest |
| `examples/hexagon/llama_cpp_integration/tl_ggml_qgemv.cc` | model registry/dispatch，调用生成 kernel |
| `examples/hexagon/llama_cpp_integration/ggml-hexagon-q8.patch` | llama.cpp CMake + Q8 dot seam |
| `testing/python/hexagon/test_tilelang_qgemv.py` | layout contract 与离线 codegen 测试 |

## 5. 模型集成边界

真实 decode 调用链：

```text
hvx_mm_matmul
  -> hvx_mv_2d_repacked_q8_0            # six workers
     -> dma_queue_pop                   # [32,K] weight ready in VTCM
     -> tiled_vec_dot_q8_0_32x1
        -> tl_dispatch(TL_OP_QGEMV_DOT)
           -> generated qgemv_q8_0_k{2048,8192}_kernel
              -> tl_hexagon_q8_0_dot_32x1
```

不匹配时 `tl_dispatch` 返回 `-1`，立即执行原 stock dot。当前例子只接受：

- Q8_0；
- `valid_rows == 32`；
- K=2048 或 K=8192。

它覆盖 standalone projection、`MUL_MAT+ADD` 的 down projection、short-conv projection 和
`lm_head`。融合 gate+up 主要走 `32x2` atom，尚未替换；这也是下一步最有价值的扩展。

## 6. 设备结果

### 6.1 atom 数值

| atom | max abs error | max relative error | 结果 |
|---|---:|---:|---|
| Q8_0 `32x2048` | 7.62939e-6 | 8.20345e-7 | PASS |
| Q8_0 `32x2048`, no bias | 3.81470e-6 | 6.53064e-7 | PASS |
| Q8_0 `32x8192` | 1.90735e-5 | 2.08145e-6 | PASS |

golden 使用逐 K=32 block 的 int32 dot、fp16 scales、fp32 accumulation 和 fp32 bias。

### 6.2 LFM2-1.2B Q8_0 模型 A/B

命令：`llama-bench -p 1024 -n 128 -r 3`，host 库允许 65536-row `lm_head` 进入 HTP。

| skel | pp1024 | tg128 |
|---|---:|---:|
| TileLang Q8 `32x1` atom | 3300.53 ± 63.94 | **35.67 ± 0.37** |
| stock Q8 atom | 3331.28 ± 20.96 | **35.46 ± 0.17** |

固定 seed、prompt `The capital of France is` 的 8-token 输出两者完全一致：

```text
Paris is the capital of France. It
```

反汇编确认 `tiled_vec_dot_q8_0_32x1` 调用 `tl_dispatch`，`tl_qgemv_run` 再调用生成的
`qgemv_q8_0_k2048_kernel`。结论是 **模型集成和性能 parity 已完成**；当前结果不支持“新 atom 比
stock 更快”的说法，也不应为了制造数字而重复手写 stock 已有的 DMA pipeline。

设备默认 `libggml-htp-v79.so` 已恢复为 stock；验证产物保存在同目录
`libggml-htp-v79.so.tilelang-q8`。

## 7. 复现

Standalone：

```bash
/home/xwh/miniforge3/envs/tl/bin/python \
  examples/hexagon/example_qgemv_q8_0.py --k 2048
/home/xwh/miniforge3/envs/tl/bin/python \
  examples/hexagon/example_qgemv_q8_0.py --k 8192
```

重新生成 model artifacts：

```bash
/home/xwh/miniforge3/envs/tl/bin/python \
  examples/hexagon/llama_cpp_integration/emit_qgemv_q8_0.py
```

在 clean llama.cpp tree 中复制三个生成/集成文件，应用 patch，然后构建 `htp-v79`：

```bash
cp examples/hexagon/llama_cpp_integration/tl_ggml_qgemv.cc \
   examples/hexagon/llama_cpp_integration/kernel_qgemv_q8_0_k2048.c \
   examples/hexagon/llama_cpp_integration/kernel_qgemv_q8_0_k2048_nobias.c \
   examples/hexagon/llama_cpp_integration/kernel_qgemv_q8_0_k8192.c \
   examples/hexagon/llama_cpp_integration/kernel_qgemv_q8_0_k8192_nobias.c \
   /path/to/llama.cpp/ggml/src/ggml-hexagon/htp/
git -C /path/to/llama.cpp apply \
   /home/xwh/tilelang-hexagon/examples/hexagon/llama_cpp_integration/ggml-hexagon-q8.patch
/home/xwh/.local/cmake/bin/cmake --build /path/to/build-snap --target htp-v79 -j8
```

## 8. 后续顺序

1. 实现 `Q8GemvIntrinEmitter.dot_32x2`，接管融合 gate+up；这是当前 Q8 profile 中最大的 decode 项。
2. 给 Q8 profile 加 AXI read/DMA wait 事件，并做纯 DMA bandwidth microbenchmark；先有测量再决定是否改 prefetch。
3. 把 software pipeline 的 commit/wait 变成 target hook，再实现 Hexagon 1D/2D async copy lowering。
4. 在 L3 完成后实现完整 semantic QGEMV op，让 layout/prefetch/worker 数可 autotune，而不是继续扩展手工 model patch。
5. 单独整理并上游化 65536 `lm_head` 支持；这是已证实的约 27% Q8 decode 收益，和 TileLang atom 优化是两件事。

设计判据很简单：TileLang 应拥有可组合的算子语义和调度，Hexagon backend 应拥有指令选择、DMA
lowering 与资源生命周期；llama.cpp 只提供资源租约和稳定的 op seam。任何一层跨过这条边界，都应
先用模型 A/B 证明必要性。
