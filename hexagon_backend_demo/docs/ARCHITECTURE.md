# TileLang 适配 Qualcomm Hexagon NPU：架构与抽象参考

项目总体技术路线、实施阶段和验收标准见 [`TECHNICAL_ROUTE.md`](TECHNICAL_ROUTE.md)。本文聚焦已经
实现的硬件映射和代码结构。

## 1. 目标与边界

目标不是为某个模型写一组固定 kernel，而是让 TileLang 获得一个可扩展的 Hexagon backend：

1. 普通 TileLang PrimFunc 可以 lower 成 Hexagon C；
2. `T.alloc_shared`、`T.copy`、`T.gemm`、`T.Layout` 和 `T.Kernel(num_workers=N)` 有明确硬件映射；
3. 无法由通用算术可靠恢复的 HVX/HMX 指令以 intrinsic atom 暴露；
4. kernel 可以独立 FastRPC 运行，也可以嵌入 llama.cpp 等已有 DSP skel；
5. 新算子沿相同层次扩展，而不是把调度、资源所有权和指令语义塞进一个 monolithic tile-op。

Q4 matmul 和 Q8 GEMV 是两种不同数据流的验证案例：前者验证 HMX prefill、布局和并行 dequant，
后者验证 HVX decode dot 以及复用宿主流式 pipeline。

## 2. 硬件模型

| 资源 | v79 特征 | Backend 含义 |
|---|---|---|
| HVX | 1024-bit / 128B 向量，6 个 context | 混合 dtype 必须按完整寄存器宽度 vectorize；适合 dequant、copy、reduce、GEMV |
| HMX | 单个 FP16 32x32 Crouton 矩阵引擎 | 适合 M/N 有复用的 matmul；累加器隐式且全核唯一，需要显式状态协议 |
| VTCM | 8 MiB 软件管理 SRAM | `alloc_shared` 的落点；必须显式规划对齐、生命周期和 worker 分区 |
| DMA | 支持 ordered async 1D/2D 搬运 | 长期应由 target-specific async-copy lowering 管理；嵌入模式可借宿主队列 |
| FastRPC | ARM host 到 cDSP user PD | standalone runtime 需要生成 IDL/skel/host；嵌入模式必须复用宿主 session |

Hexagon 不是 SIMT GPU。Backend 不能假设 warp、自动 cache、可寻址 tensor-core accumulator 或
进程内 kernel launch。

## 3. 五层结构

```text
L1  TileLang DSL / PrimFunc
    T.Kernel, T.copy, T.gemm, T.Layout, explicit intrinsic atoms
                     |
L2  Hexagon TIR pipeline
    CPU-like lowering + Hexagon tile-op dispatch + memory/liveness passes
                     |
L3  CodeGenTileLangHexagon
    TIR -> plain C, serialized grid, VTCM offsets, worker entry ABI
                     |
L4  Hardware recipe templates
    HVX/HMX/VTCM/worker/Q4/Q8 instruction sequences
                     |
L5  Runtime and embedding
    standalone FastRPC adapter OR host-owned Mode B resource lease
```

主要原则是：L3 保持薄，具体指令和资源配方放在 L4；调度组合留在 L1/L2；资源的申请和释放由
L5 的实际 owner 负责。

## 4. 通用 DSL 映射

| TileLang | Hexagon lowering |
|---|---|
| `T.Kernel(grid)` | 单 DSP 线程上的显式 block loops |
| `T.Kernel(..., num_workers=N)` | 外层 block 以 stride 分配给 persistent worker pool |
| `T.alloc_shared` | VTCM arena 中的对齐静态 offset |
| `T.copy` | 可向量化的 HVX load/store；异步 DMA 是待完善 target hook |
| `T.gemm` | 高层路径调用 `tl_hexagon_hmx_gemm`；低层路径使用显式 HMX atoms |
| `T.Layout` | activation/weight/output Crouton 的逻辑到物理地址映射 |
| `T.call_extern` atom | 无法从普通 TIR 稳定恢复的 HVX/HMX 指令序列 |

Backend 同时保留两种 matmul 抽象：

- **高层、通用路径**：用户写 `T.gemm`，runtime template 负责 Crouton pack、HMX MAC 和 unpack；
- **低层、可调度路径**：`HMXIntrinEmitter` 暴露
  `acquire/load_bias/clear/mma/convert/store/release`，用户控制 K loop、layout 和融合点。

两者不是互斥的。普通算子优先使用高层接口；量化融合或特殊流水线才下降到 instruction atom。

## 5. TileLang 到设备的完整流程

```text
@T.prim_func
  -> TileLang/TVM TIR
  -> resolve_pipeline("hexagon")
  -> LowerTileOp + VTCM/liveness/layout transforms
  -> SplitHostDevice, preserve host declaration parameter order
  -> CodeGenTileLangHexagon
  -> generated C + selected tl_templates/hexagon headers
  -> hexagon-clang++ -mhvx -mhmx
  -> FastRPC skel.so
  -> adb deploy -> cDSP user PD
  -> torch-callable HexagonKernelAdapter
```

三个入口分别适合不同阶段：

- `tilelang.lower(..., target="hexagon")`：检查生成 C 和指令 atom，不连接设备；
- `tilelang.compile(...)` / `@tilelang.jit`：生成 FastRPC 工程、构建、部署并执行；
- embeddable emitter：只输出裸 device function 和 manifest，供已有 runtime co-compile。

## 6. Standalone 与嵌入模式

### Mode A：Standalone FastRPC

TileLang runtime 自己生成 IDL、ARM host、DSP skel 和 CMake，申请 VTCM/HMX，部署后通过持久 agent
调用。它是 backend 开发、数值验证和 microbenchmark 的首选模式。

### Mode B：Embedded Host Skel

llama.cpp 已拥有 FastRPC session、VTCM、DMA、worker pool 和 HMX compute resource。TileLang kernel
作为源码 co-compile 到该 skel，只接收一个窄资源租约：

```text
tl_op_ctx = tensors + shape + VTCM region + HMX handle
          + ordered DMA callbacks + synchronous parallel callback
```

TileLang 不创建第二个 session 或 worker pool。`matches/run` 返回失败时宿主继续 stock kernel。
这套 registry/lease 模式可以复用于其他推理 runtime，不绑定 llama.cpp。

当前集成的自动化边界必须明确：TileLang 自动生成 device `.cc` 和 manifest；framework adapter、
registry wiring 与 CMake source list 仍是显式代码。完整的文件落点、build 命令、dispatch 调用链和
fallback 见 [`LLAMA_CPP_INTEGRATION.md`](LLAMA_CPP_INTEGRATION.md)。

## 7. 案例如何验证通用抽象

### 普通 FP16 matmul

验证 `T.alloc_shared -> VTCM`、`T.copy -> HVX`、`T.gemm -> HMX` 和完整 standalone runtime。

### Worker pool

验证 `num_workers`、per-worker VTCM、6 路 HVX 并行，以及单 HMX accumulator 的串行保护。

### Q4 matmul

一块 576B Q4_0 tile 由 native HVX atom 解码为 32x32 FP16 weight Crouton；TileLang 生成 pack、
parallel dequant range 和显式 HMX compute 三个 stage。llama.cpp 只安排 2D DMA、VTCM 和 worker。

### Q8 GEMV

TileLang 生成 `32x1` signed-int8 HVX dot atom，并替换宿主已经 DMA staged 的 tile。该案例说明
instruction atom 可以复用成熟宿主 pipeline，而不重复实现完整 operator scheduler。

## 8. 扩展一个新算子的步骤

1. 定义算子语义、dtype、layout 和 fallback contract；
2. 判断普通 TIR/vectorization 是否足够；不够时定义最小硬件 atom；
3. 用 `T.Layout` 表示稳定地址 ABI，用 TileLang loop 表示可调调度；
4. standalone lower 和 golden correctness；
5. 在真实设备验证 VTCM、对齐、worker 和错误传播；
6. 需要框架集成时生成 embeddable body + manifest；
7. 宿主只提供资源租约和 dispatch seam；
8. 以算子级 profile 和模型 A/B 同时验证，不从单次 FastRPC 墙钟推断性能。
