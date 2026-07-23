# 项目评估与任务列表

## 1. 总体判断

当前成果已经证明 TileLang 可以作为 Qualcomm Hexagon 的真实 backend：通用 FP16 matmul、HVX
elementwise/reduction、VTCM、persistent worker pool、standalone FastRPC、低层 HMX emitter、Q4/Q8
量化 atom 和 llama.cpp 嵌入均已在 v79 设备运行。

它仍是研究级 backend，不是生产级产品。已验证能力集中在 v79 和有限 operator/shape；通用异步 DMA
lowering、autotune、多版本硬件验证、可重入 runtime context 和持续集成仍需完成。

## 2. 能力评估

| 维度 | 评分 | 依据 |
|---|---:|---|
| 端到端功能完整度 | 8/10 | DSL -> codegen -> build -> deploy -> execute，以及 embedded 模式均打通 |
| 抽象边界 | 7/10 | DSL/layout/atom/resource lease 已分层；async copy 仍借宿主回调 |
| 选定算子性能 | 8/10 | HMX matmul、Q4 prefill、Q8 dot 已达到 stock 同档 |
| Operator 覆盖 | 5/10 | matmul/flash-attn/rmsnorm/Q4/Q8 已有；完整算子库不足 |
| 硬件版本可移植性 | 5/10 | 设计可参数化，但主要只在 v79 真机验证 |
| 生产可用性 | 4/10 | 缺 CI、稳定 ABI、autotune、资源 context 化和发布流程 |

## 3. 工作包状态

| ID | 工作包 | 状态 | 证据/剩余问题 |
|---|---|---|---|
| BE-01 | Hexagon target 注册与 build dispatch | 完成 | `target.build.tilelang_hexagon` 和 kernel cache 已接通 |
| BE-02 | Hexagon TIR pipeline 和 CPU-like tile-op lowering | 完成 | copy/fill/gemm/reduce 均有 target dispatch |
| BE-03 | HVX 全寄存器 vector codegen | 已验证 | 1024-bit 宽度和 mixed-dtype 约束已进入 codegen |
| BE-04 | VTCM memory scope、对齐与 high-water | 已验证 | standalone 和 worker slice 均通过设备测试 |
| BE-05 | 高层 `T.gemm -> HMX` | 已验证 | 普通 matmul 和 flash attention 可运行 |
| BE-06 | `HMXIntrinEmitter` + `T.Layout` | 已验证 | 显式 accumulator protocol 和 Crouton layouts |
| BE-07 | Persistent 1-HMX/6-HVX worker pool | 已验证 | per-worker VTCM，HVX workload 可见加速 |
| BE-08 | Standalone FastRPC runtime/agent | 已验证 | 自动 IDL/skel/host/build/deploy/marshal |
| BE-09 | Embedded registry 与资源租约 | 实验完成 | llama.cpp 验证；需要 context 化与稳定 ABI |
| BE-10 | 通用 async DMA / software pipeline lowering | 部分完成 | 嵌入模式已有 1D/2D callback；DSL target hook 尚未完成 |
| BE-11 | Q4/HMX 案例 | 已验证 | native tile atom + parallel staged schedule + 模型 A/B |
| BE-12 | Q8/HVX GEMV 案例 | 已验证 | `32x1` atom 和模型 parity；`32x2` 未覆盖 |
| BE-13 | llama.cpp 框架集成 | 实验完成 | generated TU、adapter、CMake、fallback 和模型运行均可演示 |
| BE-14 | Autotune 和 schedule search | 未开始 | tile size、worker、prefetch、VTCM plan 仍手工选择 |
| BE-15 | v73/v75/v81 多版本验证 | 未完成 | 编译参数存在，缺真实设备/CI 验证矩阵 |
| BE-16 | 生产化发布与上游整理 | 未完成 | 需拆分变更、稳定测试和维护边界 |
| BE-17 | Manifest-driven embedding 工具 | 未完成 | manifest 尚未自动生成 registry/CMake；Q4/Q8 registry 仍需合并 |

## 4. 已达成的验收项

- 同一套 TileLang `T.gemm` 可以 lower 并在 Hexagon HMX 执行；
- generated C 中可以检查 VTCM offset 和 HMX calls；
- standalone kernel 对 torch golden 数值正确；
- worker pool 在真实 6-context HVX 上运行；
- instruction emitter 不依赖 `T.gemm`，可以表达 HMX 隐式 accumulator；
- Q4/Q8 量化格式拥有 layout 和硬件 atom，而非 opaque 完整 operator；
- generated body 可以 co-compile 进入 llama.cpp DSP skel；
- 不匹配路径安全回落 stock；
- Q4/Q8 模型输出与 stock 一致，选定路径性能达到同档；
- 关键源码、生成文件、skel、日志和演示入口已归档到本目录。

## 5. 后续优先级

### P0：把研究 backend 变成稳定 backend

1. **BE-10 async copy target hook**：把 pipeline commit/wait 从 PTX 硬编码抽成 target interface，
   实现 Hexagon 1D/2D DMA lowering 和事件依赖；
2. **Runtime context 化**：移除 HMX/VTCM/registry 的进程级 static ownership，使多个 embedded op
   和多个 session 可重入；
3. **测试矩阵**：离线 lowering、Hexagon compile-only、设备 smoke、fallback 和错误传播分层；
4. **性能测量 API**：统一 qtimer/PMU/DMA wait，避免 FastRPC marshal 污染 kernel 数据。
5. **Embedding 打包**：由 manifest 生成 registry/source list，提供稳定的 CMake helper，并允许多个
   TileLang op 共存于同一 DSP skel。

### P1：提高覆盖与可调度性

1. 把 worker 数、tile shape、prefetch 深度和 VTCM plan 接入 autotune；
2. 完成 Q8 `32x2`、larger-M staged HMX、bias/epilogue 和 tail shapes；
3. 提供语义级 quantized matmul/GEMV op，再由 layout/atom/schedule 分层 lowering；
4. 扩充 reduction、normalization、attention 和 convolution 的 idiomatic TileLang 路径。

### P2：泛化框架与硬件

1. 把 `tl_op_ctx` 资源租约抽成独立 embedding ABI，而不是放在 llama.cpp 示例中；
2. 在 v73/v75/v81 设备验证 feature gating 和 template 分支；
3. 接入其他 Qualcomm runtime/framework，证明 Mode B 不依赖 ggml；
4. 制定可发布的 backend 包、SDK 探测和兼容性说明。

## 6. 不应继续采用的路线

- 为每个量化格式新增一个包含 DMA、线程、dequant 和 compute 的 monolithic `T.gemm` 变体；
- 在 embedded kernel 内再次申请 FastRPC session、VTCM 或创建 worker pool；
- 把 DMA descriptor 位域直接暴露到用户 DSL；
- 只看 standalone 单次墙钟或模型 tok/s，不做算子 profile 和 stock 同口径 A/B；
- 因某个固定模型 shape 有效，就把 shape 常量固化成通用 backend API。
