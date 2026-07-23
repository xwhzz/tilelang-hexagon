# 演示讲稿

## 1. TileLang × Qualcomm Hexagon NPU Backend

开场给出主线：typed primitives、手写性能上限、explicit TileLang、通用调度、framework package。
案例是验证材料，不定义 backend。

## 2. 项目目标与验收边界

验收包括可读的 generated code、真实设备正确性、可扩展的 operator 路径和安全框架 fallback。
Primitive 盘点与 workload profiling 并行推进。

## 3. Hexagon 硬件模型

Hexagon 不是 SIMT GPU：HVX 是全宽向量，HMX 使用隐式 accumulator，VTCM 和 DMA 由软件管理，
设备通过 FastRPC 运行。这些事实决定 backend 抽象。

## 4. 抽象边界与所有权

Operator semantics、TileLang schedule、typed primitives、data movement 和资源生命周期分层。
Standalone 由 TileLang 持有资源；embedded 由宿主持有。

## 5. 五层 Backend 架构

Codegen 保持为薄 C emitter。硬件类型和指令 recipe 集中在 Hexagon target templates，runtime 只负责
资源、构建和执行。

## 6. TileLang 到 Hexagon 的生成流程

同一个 PrimFunc 可以停在 generated C 检查、包装为 standalone FastRPC 工程，或输出 embeddable
device body 和 manifest。

## 7. 高层 T.gemm 的目标形态

`T.gemm` 是产品最终入口，不是 backend 的第一阶段实现方法。先用 explicit primitives 对齐硬件和
性能，结构稳定后再上提为高层 recipe。

## 8. 低层 HMX 与 Layout 抽象

HMX 像 GPU MMA 一样暴露最小 atoms，但保留真实差异：accumulator 不可寻址，Crouton 是物理 layout，
convert/store 和依赖必须显式。

## 9. Standalone 与 Embedded

两种模式复用 generated body，但资源 owner 不同。Embedded kernel 不能再次申请 session、VTCM、
worker 或 HMX resource。

## 10. 并行、VTCM 与 DMA

六个 HVX context 可以并行数据处理，但 HMX 仍是单资源。`T.Pipelined` 表达依赖，Hexagon lowering
再选择 ordered DMA 或 HVX fallback。

## 11. HMX 纵向案例

这一页只用于说明 layout、最小数据转换 atom、worker 分工和显式 HMX K-loop 如何组合，不作为
backend API 设计依据。

## 12. 最小 llama.cpp 集成

先固定一个 dtype 和 shape，不引入通用 registry。TileLang 生成普通 C ABI 函数，translation unit
直接编进现有 DSP skel；stock op 前增加一次调用，未命中或失败继续原实现。

## 13. Primitive-first 技术路线

顺序是 catalog、handwritten ceiling、explicit TileLang、generic schedule、framework package。
Profiling 并行决定首批热点算子。

## 14. 当前 Primitive 缺口与 FP32 输出

近期阻塞是 typed HVX。FP32 add 的 qf32 结果必须转换为 IEEE sf 后存储；HMX 还需要 typed output
contract 和 completion dependency。

## 15. 重点任务与执行顺序

先补 typed primitives 和 conformance tests，再建立手写基线，然后追 explicit TileLang parity。
框架 registry 和 resource lease 作为并行线从一开始共用。

## 16. 现场演示流程

优先演示 lower、codegen 和普通设备路径；有设备时再展示 framework dispatch。结束后确认恢复 stock
skel。

## 17. 结论

最终交付不是固定算子集合，而是 typed primitives、可组合 schedule、可验证 runtime contract 和
manifest-driven framework package。
