# TileLang Qualcomm Hexagon Backend 技术路线

> 主文档。技术路线采用 **primitive-first、性能基线驱动、框架接入并行推进** 的顺序。
> 具体数据格式和单一 workload 只用于验证，不定义 backend 抽象。

## 0. 一页结论

建议把项目组织成五条有依赖关系的工作线：

```text
硬件 primitive 盘点与类型语义 -----------------------------+
       |                                                     |
       v                                                     v
手写 HVX/HMX kernel -> 硬件性能上限 -> 显式 TileLang kernel -> 通用调度优化
       ^                                  |                  |
       |                                  v                  v
workload profile -> 热点算子/融合候选       AOT package -> llama.cpp dispatcher
```

其中：

1. **Primitive 是第一优先级**：先把 HMX、HVX、DMA、worker 和 VTCM 的最小硬件语义绑定完整；
2. **类型语义先于算子**：输入 dtype、计算表示、指令结果表示和存储 dtype 必须分开；
3. **手写 kernel 是性能标尺**：先确认硬件和算法能够达到目标，再要求 TileLang 复现；
4. **TileLang 先显式、后自动**：先暴露类似 GPU MMA 的 atoms，不急于封装成大 `T.gemm`；
5. **通用优化最后接入**：显式 kernel 达到手写基线后，再启用 layout、pipeline、worker 和 autotune；
6. **框架接入并行推进**：手写 kernel 和 generated kernel 使用同一 ABI、registry 和 fallback；
7. **Profiling 决定算子顺序**：不预设某个算子重要，按真实调用占比、shape 和可融合流量选择。

这条路线的关键不是“先写很多 extern 函数”，而是建立一个有类型、有副作用描述、有测试和性能证据的
Hexagon primitive 层。

## 1. 抽象边界

### 1.1 五层对象不能混在一起

| 层次 | 含义 | 应包含 | 不应包含 |
|---|---|---|---|
| Hardware primitive | 单条指令或不可拆的短序列 | dtype、lanes、结果表示、side effect | tile traversal、完整算子 |
| Hardware recipe | 少量 primitives 的稳定组合 | Crouton pack、转换、reduction tree | framework tensor、session 管理 |
| TileLang schedule | 数据流和执行顺序 | loop、tile、layout、pipeline、worker | SDK descriptor 位域 |
| Operator | 可验证的计算语义 | shape、tail、epilogue、融合边界 | FastRPC/VTCM ownership |
| Framework adapter | 宿主 ABI 和 dispatch | tensor view、resource lease、fallback | HMX/HVX 指令选择 |

判断一个接口是否应该是 primitive，可以使用四条规则：

1. 是否对应一条硬件指令或必须连续发射的短序列；
2. 是否有稳定、可独立测试的输入输出语义；
3. 是否不拥有 operator loop 和资源生命周期；
4. 是否可能被多个算子 schedule 复用。

例如 FP32 vector add 可以是 primitive；完整 normalization、attention 或 convolution 不应是 primitive。

### 1.2 建议的实现分层

```text
TileLang semantic expression / explicit emitter
                |
                v
typed TIR intrinsic or target primitive id
                |
                v
thin Hexagon codegen mapping
                |
                v
tl_templates/hexagon typed wrapper
                |
                v
Q6_* intrinsic / HMX asm / runtime callback
```

`tl_templates/hexagon` 可以隐藏硬件返回表示的必要转换，但不能隐藏 operator schedule。例如
FP32 add 的 qf32-to-sf 转换属于 primitive contract；遍历整行、做 residual add 再写回不属于 primitive。

## 2. 当前 primitive 底座与主要缺口

### 2.1 HMX 当前状态

当前已有：

- accumulator 的 acquire/release dependency；
- bias/scale state load；
- clear、32x32x32 MMA、convert、store；
- activation/weight/output Crouton `T.Layout`；
- dependency-only state token，避免把隐式 accumulator 假装成普通 buffer；
- FP16 operand 路径和真实设备验证。

主要缺口：

| 缩写 | 缺口 | 为什么重要 |
|---|---|---|
| HMX-TYPE | `convert(mode=2)` 仍是 magic number | 应明确 accumulator、convert result、rounding 和 output dtype |
| HMX-OUT | output mode/store contract 未类型化 | 无法可靠扩展不同输出类型和 epilogue |
| HMX-SYNC | convert/store completion 与 operand lifetime 仍偏手工 | pipeline 和 buffer reuse 需要可靠 dependency |
| HMX-PACK | pack/unpack 仍分散在 runtime 或算子 helper | 应沉淀为可复用的 layout/data-movement recipe |
| HMX-TAIL | native tile 之外的 padding/tail contract 不统一 | operator family 不能只支持整 tile shape |
| HMX-FEATURE | version/capability gating 不完整 | 不同 Hexagon 版本不能共享硬编码假设 |

HMX 的方向基本正确：继续补齐类型、输出模式和依赖语义，不需要先增加更大的 HMX operator。
这里必须区分 HMX convert/store 的硬件输出格式与算子的最终逻辑 dtype。如果硬件先写出 FP16 Crouton，
最终 FP32 tensor 应由后续 HVX unpack/convert recipe 产生，不能把它描述成 HMX 直接支持任意输出 dtype。

### 2.2 HVX 当前状态

当前 `hvx_math.h` 已有：

- 128B aligned/unaligned load/store；
- FP32 splat、add、sub、mul；
- FP16/FP32 widen/narrow；
- exp2、reciprocal、rsqrt recipe；
- FP16 row 到 FP32 scalar 的 sum/max；
- 一个针对连续 FP16 输出 loop 的 elementwise codegen fast path。

这里存在当前最直接的架构缺口：codegen 的内部值固定为 `HvxLanes{lo, hi}`，表示“64 个 FP16
元素 widen 成两组 32-lane FP32”；它只接受 FP16 load、FP16 store、64 的倍数，并在结尾固定 narrow
回 FP16。因此底层虽然已有 FP32 add wrapper，普通 FP32 输入或 FP32 输出仍不会进入这条通用路径。

### 2.3 FP32 add 为什么需要单独处理输出表示

当前 FP32 add wrapper 的真实语义是：

```text
IEEE sf inputs
  -> Q6_Vqf32_vadd_VsfVsf
  -> internal qf32 result
  -> Q6_Vsf_equals_Vqf32
  -> IEEE sf vector
```

所以不能把“表达式 dtype 是 float32”直接等同于“硬件指令结果已经是可存储 FP32”。至少需要区分：

| 输入存储 | 计算表示 | primitive 结果 | 输出存储 |
|---|---|---|---|
| FP32 | FP32 | qf32 -> sf | 一个 32-lane FP32 store |
| FP16 | FP32 | 两个 32-lane sf | narrow 后一个 64-lane FP16 store |
| FP16 | FP32 | 两个 32-lane sf | 两个 32-lane FP32 store |

建议把 HVX codegen 内部值改成类型化对象，而不是固定的 lo/hi pair：

```text
HvxValue
  logical_dtype
  compute_dtype
  hardware_repr      sf / qf32 / hf / integer / predicate
  logical_lanes
  register_count
  registers[]
```

Cast 也不能继续一律透明。它必须决定 conversion、rounding、saturation、register count 和最终 store。

### 2.4 DMA、worker 和 VTCM 当前状态

| 能力 | 当前状态 | 缺口 |
|---|---|---|
| VTCM allocation/alignment | 已有 shared arena 和 high-water | 缺静态 legality、lifetime 和 double-buffer planner |
| Persistent workers | runtime 已能执行同步 parallel callback | 缺统一 barrier/event 和 TileLang schedule contract |
| Ordered DMA 1D/2D | embedded ABI 已能借用宿主 callback | 缺 target-neutral TIR primitive 和 pipeline lowering |
| HMX/worker 协作 | 有实验路径 | 缺统一 dependency graph 和资源占用检查 |

这些不是普通算术 primitive，但它们是 schedule primitive；没有它们，手写 pipeline 无法干净迁移回
TileLang。

## 3. 需要对接的 primitive 清单

不建议一次性绑定 SDK 中所有 intrinsic。先建立 primitive catalog，再按 operator profile 扩展闭包。

每个 catalog entry 至少记录：

```text
primitive_id
target_features / minimum_arch
input storage dtype and hardware representation
output hardware representation and legal storage dtype
logical lanes / register count
alignment and memory scope
rounding / saturation / NaN behavior
side effects and dependency tokens
scalar or software fallback
compile test / device golden / microbenchmark
```

### 3.1 P0：HVX 基础集合

| 类别 | P0 primitives | 当前判断 |
|---|---|---|
| Memory | aligned/unaligned load/store、splat | 有基础实现，需按 dtype 类型化 |
| Convert | FP16 <-> FP32、integer widen/narrow | FP16/FP32 部分存在，contract 不完整 |
| FP32 arithmetic | add/sub/mul/fma、min/max、abs/neg | add/sub/mul 已有 wrapper，其余缺失 |
| Predicate | compare、select、mask、clamp | 通用绑定缺失 |
| Reduction | FP16/FP32 sum/max/min | 只有受限 FP16 row sum/max |
| Tail | masked load/store 或明确 scalar tail | 只有部分 helper 自行 fallback |

P0 完成后，最小验收应包括：

- `fp32 + fp32 -> fp32` vector add；
- `fp16 + fp16 -> fp16`，内部允许 FP32 compute；
- `fp16 + fp16 -> fp32`；
- aligned、unaligned、整向量和 tail；
- generated source 不出现逐元素 half conversion libcall。

### 3.2 P1：HVX 算子支撑集合

| 类别 | primitives/recipes | 主要用途 |
|---|---|---|
| Integer | i8/u8/i16/i32 add/sub/mul、bitwise、shift | unpack、index、数据变换 |
| Dot/MAC | integer dot、widening MAC、horizontal accumulate | vector dot 和小矩阵路径 |
| Permute | shuffle、interleave、deinterleave、rotate、table lookup | layout、RoPE、pack/unpack |
| Math recipe | reciprocal、rsqrt、exp2，再组合 sigmoid/tanh | activation、normalization、softmax |
| Reduction recipe | tree reduction、multi-row reduce、argmax | normalization、attention、sampling |

`sigmoid`、`SwiGLU` 或完整 softmax 应作为 recipe/operator，不应伪装成单条硬件 primitive。

### 3.3 P0：HMX 集合

| 类别 | primitives/contract | 当前判断 |
|---|---|---|
| Resource | thread enable、exclusive accumulator acquire/release | runtime 与 emitter 已有基础 |
| State | clear、bias/scale load | 已有基础 |
| Matrix | native MMA atom | 已有并已验证 |
| Convert | typed convert mode、rounding、scale/bias | 需要替换 magic mode |
| Store | typed store、completion token | store 已有，类型和依赖需补齐 |
| Layout | activation/weight/output Crouton layout | 已有，需 feature/version metadata |
| Data movement | typed pack/unpack、FP32/FP16 boundary | 有零散 helper，需通用化 |

### 3.4 P0：Schedule/runtime 集合

- `dma.copy_1d`、`dma.copy_2d`、`dma.wait` 或等价 completion token；
- worker launch、worker id、barrier、join；
- VTCM region、alignment、lifetime 和 per-worker slice；
- HMX stage 与 HVX/DMA stage 的依赖；
- qtimer/PMU region marker，仅用于 profiling，不进入算子语义。

## 4. 推荐实施顺序

### M0：冻结 primitive contract，同时准备 workload profile

任务：

1. 建立上述 catalog 和 capability key；
2. 把 logical dtype、compute dtype、hardware repr、storage dtype 分开；
3. 记录现有 primitives 的真实输入输出和缺失测试；
4. 定义手写 kernel 与 TileLang kernel 共用的 benchmark ABI；
5. 定义 workload profiling 输出格式，用于后续选择算子和融合候选。

退出标准：任何新 primitive 都能回答“输入是什么、硬件返回什么、如何存储、如何测试”。

### M1：完成 P0 primitives

优先顺序：

1. 类型化 HVX load/store 和 FP32 vector add；
2. FP16/FP32 conversion 与三种输出组合；
3. FMA、min/max、predicate/select 和 reduction；
4. HMX typed convert/store 与 completion dependency；
5. DMA/worker/VTCM schedule primitives；
6. compile-only、device golden 和 instruction inspection 测试。

退出标准：P0 catalog 全部有独立测试；不依赖完整模型即可验证正确性和指令选择。

### M2：用手写 kernel 建立性能上限

从 profile 选出的热点中，每类只选一个代表性算子：

1. 纯 HVX elementwise/residual；
2. map + reduction 的 normalization；
3. gated activation；
4. HMX projection/matmul；
5. HMX + HVX + DMA 的 tiled pipeline；
6. 需要滑窗或 permute 的算子。

手写版本必须使用与未来 generated kernel 相同的输入 layout、VTCM 预算、worker 数和 runtime ABI。
每个版本记录 compute、load/store、DMA wait、pack/unpack 和同步成本。

退出标准：明确硬件上限和主要瓶颈。如果手写版本仍不能达到目标，应先调整 primitive、layout 或算法，
而不是立即增加 compiler pass。

### M3：显式 TileLang kernel 复现手写调度

迁移顺序：

```text
handwritten intrinsic C
  -> explicit TileLang primitive calls + explicit loops
  -> T.Layout 表达物理地址
  -> TileLang worker/VTCM/DMA schedule
  -> generated C/assembly 与手写版本对照
```

这一阶段不要求使用高层 `T.gemm`。HMX 应像 GPU MMA 一样允许显式 acquire/clear/MMA/convert/store；
HVX 也需要显式 load/arithmetic/convert/store emitter，供性能对齐和复杂融合使用。

退出标准：显式 TileLang kernel 正确，且核心阶段性能与手写基线的差距在预先约定范围内；初始建议
以 10% 作为需要解释的阈值，而不是硬性产品指标。

### M4：接入 TileLang 通用优化

按一次只引入一个变量的顺序验证：

1. `T.Layout` 与 layout inference；
2. vectorization 和 typed HVX expression lowering；
3. `T.Pipelined` producer/consumer stage；
4. ordered DMA selection、double buffer 和 wait placement；
5. `num_workers`、worker specialization 和 VTCM slice；
6. loop unroll、fusion boundary 和 common subexpression reuse；
7. tile、pipeline depth、worker 数和 VTCM plan autotune。

每项优化都与 M3 的显式版本比较。性能下降时必须能定位是 codegen、layout、同步还是资源规划，不能只看
端到端墙钟。

### M5：形成高层 recipe 和 operator family

只有当多个显式 kernel 共享相同稳定结构时，才上提为：

- `T.copy` 的 Hexagon DMA/HVX recipe；
- `T.gemm` 的 HMX recipe；
- 通用 reduction/map recipe；
- normalization、attention、convolution 等 operator schedule family。

高层接口必须保留下降到 explicit primitive 的路径，不能把 layout、dtype conversion 或 pipeline
重新隐藏到不可调度的 monolithic extern call 中。

## 5. Workload profiling 与算子选择

Profiling 是算子选择门槛，不是 backend 正确性的替代品。建议至少输出：

- operator kind、shape、dtype、stride、调用次数和总耗时；
- warm-up、steady-state 和不同 execution phase；
- kernel time、DMA wait、marshal/runtime overhead；
- 相邻节点、共享 tensor、layout conversion 和可消除中间流量；
- stock implementation、手写 kernel 和 TileLang kernel 的同口径数据。

候选优先级不只看单次 latency，应综合：

```text
收益候选 = 总耗时占比
         x 可覆盖 shape 比例
         x 可消除内存流量
         x primitive 复用价值
         / 实现与集成复杂度
```

建议优先观察的 operator family：

1. projection/matmul 与其 bias/residual epilogue；
2. normalization + scale；
3. gated activation 及其前后 elementwise；
4. attention 内的 matrix、softmax 和 layout pipeline；
5. short convolution/SSM 类滑窗计算；
6. 高频 copy、cast、reshape materialization。

融合只在以下条件同时满足时进入实现：相邻关系稳定、dtype/layout 兼容、中间结果不被其他节点消费、
VTCM 可容纳、融合后不会破坏 fallback。融合模式属于 operator/graph 层，不属于 primitive 层。

## 6. TileLang 应提供的两条编程路径

### 6.1 普通路径

算子作者使用语义表达，compiler 选择 Hexagon primitive：

```python
for i in T.serial(N):
    C[i] = A[i] + B[i]
```

当 dtype、stride、alignment 和 tail 合法时，typed HVX lowering 选择 FP32 或 FP16 vector add；否则使用
明确的 fallback。不能再用“只识别 FP16 store”决定是否 vectorize。

### 6.2 Expert 路径

复杂融合或性能对齐允许显式组合：

```python
va = H.load(A, offset, dtype="float32")
vb = H.load(B, offset, dtype="float32")
vc = H.add(va, vb, out_dtype="float32")
H.store(C, offset, vc)
```

HMX 保持显式 protocol：

```python
acc = T.alloc_hmx_accumulator()
cvt = T.alloc_hmx_convert_state()
bias = T.alloc_hmx_bias_state()
H.acquire(acc)
H.load_bias(bias, bias_vtcm)
H.clear(acc)
for kt in T.serial(KT):
    H.mma(acc, a_hmx, b_hmx, kt)
H.convert(cvt, acc, bias, mode=H.OutputMode.FP16_CROUTON)
store_done = H.store(cvt, c_hmx, depends_on=acc)
H.release(acc, after=store_done)
P.unpack(c_hmx, C, out_dtype="float32")
```

以上 API 名称表示目标形态；真正冻结前应由 primitive catalog 决定参数和 result type。

## 7. 更干净地接入 llama.cpp

框架接入不应等所有 TileLang 优化完成，但必须与 primitive/operator 解耦：

```text
llama.cpp / ggml operator
        |
        v
build tl_op_ctx once
        |
        v
tl_dispatch(ctx) -> shared registry -> handwritten or generated implementation
        |
        +-> decline/error -> stock implementation
```

统一 package 建议包含：

```text
operator.hexagon/
├── kernel.cc
├── manifest.json
├── registry_stub.cc
├── TileLangHexagon.cmake
└── include/tl_embed.h
```

Manifest 描述 symbol、shape/dtype/stride、layout、VTCM high-water、HMX/DMA/worker requirement 和
fallback contract。宿主只负责：

- 转换 ggml tensor view；
- 借出已经持有的 VTCM、DMA、worker 和 HMX resource；
- 调用一次稳定 dispatcher；
- 在 decline 或错误时运行 stock path；
- 提供统一 profiling marker。

手写性能基线和 TileLang generated kernel 应注册到同一 registry。这样可以在不反复修改 llama.cpp
stock operator 的情况下做 A/B，也能把性能问题定位到 kernel 而不是集成代码。

## 8. 任务列表与当前评估

| ID | 任务 | 当前状态 | 下一验收点 |
|---|---|---|---|
| PR-00 | Primitive catalog/schema | 未完成 | HMX/HVX/runtime 全部登记类型与副作用 |
| PR-01 | Typed `HvxValue` 和 dtype-driven codegen | 阻塞项 | FP32 load/add/store generated path |
| PR-02 | FP16/FP32 conversion/output matrix | 部分完成 | 三种输入输出组合设备 golden |
| PR-03 | HVX FMA/min/max/predicate/select | 未完成 | elementwise primitive suite |
| PR-04 | HVX reduction 泛化 | 部分完成 | FP16/FP32 sum/max/min + tail |
| PR-05 | HVX integer/dot/permute | 零散 helper | 提炼 P1 通用 bindings |
| PR-06 | HMX state/MMA/Layout | 核心已验证 | 从案例测试拆成 primitive conformance |
| PR-07 | HMX typed convert/store | 未完成 | 删除 magic mode，覆盖合法 output dtype |
| PR-08 | HMX completion/lifetime | 部分完成 | pipeline 可证明安全复用 operand buffer |
| PR-09 | DMA/worker/VTCM schedule primitives | runtime 部分完成 | TIR contract + standalone/embedded lowering |
| PR-10 | Primitive compile/device/microbench suite | 不完整 | 每个 P0 primitive 三层测试 |
| PF-00 | Workload profile 与候选清单 | 工具已有 | 固定输出 schema 和选择规则 |
| HW-00 | 手写 primitive microkernel | 零散存在 | 统一 harness 和性能报告 |
| HW-01 | 手写 operator 性能基线 | 未系统化 | 每类一个代表性 kernel |
| TL-00 | Explicit HVX emitter | 未完成 | 与手写 vector kernel 对齐 |
| TL-01 | Explicit HMX emitter | 核心已存在 | typed output、dependency 和更多 shape |
| TL-02 | Pipeline/Layout/worker 优化 | 部分完成 | 逐项达到显式 kernel baseline |
| IN-00 | Stable embedding ABI/resource lease | 实验状态 | versioned `tl_op_ctx` |
| IN-01 | Shared registry/package/CMake | 未完成 | 第二个 op 接入不修改 stock function |
| IN-02 | Fusion dispatch contract | 未完成 | graph pattern 与 primitive/operator 分层 |

当前最重要的判断：**HMX atom 主链路已经存在，真正的近期阻塞是 HVX 类型化 primitive 层、HMX 输出
类型 contract，以及两者统一的 conformance/performance harness。**

## 9. 执行波次

### Wave 1：类型和 primitive 底座

`PR-00/01/02/06/07/10`，同时冻结 `IN-00` ABI 和 `PF-00` profiling schema。

### Wave 2：手写性能闭环

`PR-03/04/05/08/09` + `HW-00/01`，输出热点 operator 的硬件上限与 primitive 缺口。

### Wave 3：显式 TileLang parity

`TL-00/01`，逐个把手写 loop、layout、dtype conversion 和 dependency 搬进 TileLang。

### Wave 4：通用调度和融合

`TL-02/IN-02`，接入 pipeline、DMA、worker、VTCM planner 和融合 pattern。

### Wave 5：产品化接入

`IN-01`、多版本 feature gating、CI、autotune database 和可安装 package。

## 10. 验收体系

### Primitive gate

- C wrapper compile；
- generated code 命中预期 intrinsic/asm；
- device golden 覆盖正常值、边界值、alignment 和 tail；
- rounding、saturation、NaN 行为有明确 contract；
- 无意外 scalar libcall；
- 独立 microbenchmark 可重复。

### Kernel gate

- 手写与 TileLang 使用相同 ABI/layout/resource budget；
- 正确性、determinism 和 buffer lifetime 通过；
- pack/compute/DMA/sync 分段计时；
- 显式 TileLang 与手写基线的差距可解释；
- 通用优化相对显式版本有独立收益证据。

### Framework gate

- hit、decline、error 和 stock fallback 均可测；
- generated kernel 不重复 acquire 宿主资源；
- 多个注册算子共存；
- profiling 能区分 kernel 与 framework overhead；
- 移除某个 package 后 stock 行为不变。

## 11. 需要避免的路线

- 一次性绑定所有 SDK intrinsic，却没有 dtype、side effect、测试和 operator 需求；
- 把 qf32、sf、FP16 storage 当成同一种“float”处理；
- 每个算子新增一个包含 loop、DMA、worker 和 compute 的 monolithic extern call；
- 手写 kernel 尚未达到目标，就先写复杂 compiler pass；
- 显式 TileLang 尚未接近手写基线，就直接上高层 `T.gemm` 或 autotune；
- 每加入一个算子就修改一次 llama.cpp stock function；
- 只看端到端吞吐，不区分 primitive、kernel、DMA 和 framework overhead；
- 为单一 shape 固化 backend API。

## 12. 最终形态

最终 backend 应形成一条可逐层验证的梯子：

```text
typed hardware primitives
  -> reusable HMX/HVX/layout/DMA recipes
  -> explicit TileLang schedules
  -> generic pipeline and autotune
  -> operator families
  -> manifest-driven framework packages
```

新增硬件能力先进入 primitive catalog；新增算子先由 profiling 和手写 baseline 证明价值；新增 compiler
优化必须相对显式 TileLang kernel 证明收益；新增框架只实现一次 resource lease 和 dispatcher。这样才能
把“能运行的案例”逐步变成可维护、可扩展的 Qualcomm Hexagon backend。
