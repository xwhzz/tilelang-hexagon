# 前五项 DSL 算子：DMA 与多 worker 优化

显式 DMA 与多 worker 的组合包括
四项 FP32 的分块调度，以及 GEMM 的通用并行布局复制。算子语义、分块、预取和
HMX 指令顺序都位于 `examples/hexagon/example_demo_ops.py`，没有手写 C++ 行算子
或完整算子注入；使用 TileLang 自己的 DMA FIFO、worker pool、VTCM 和 FastRPC。

## 实现边界

- **DMA＋worker**：每个 tile program 创建自己的描述符和 FIFO，buffer 位于所在 worker
  的私有 VTCM arena；正常结束时排空，错误退出先收尾 DMA，再释放已持有的 HMX 单元，
  worker 的 HMX 线程使能由 runtime 在该 worker 结束时释放。
  已测 1/2/6 worker、环形队列回绕、64 KiB 块、65 元素尾部、DSP 脏输入、保护区，
  以及持有 HMX 时 DMA 队列满后在同一 session 恢复执行。
- **FP32 分块**：`make_map_tiled` 用显式双槽预取、连续分组、行内 HVX 计算组合。
  RMSNorm 保留 FP32 平方和局部 `hexreduce` macro，把每行归一化系数留在寄存器。
  优化路径要求完整 tile、N 为 32 倍数；非整齐形状保留基础实现。
- **通用复制**：修复 `vec_type` 隐式结构体复制变为逐 128-byte `memcpy` 调用的问题，
  显式复制 native vector。`T.copy` 可以融合 FP32↔FP16 与 activation/output Crouton
  布局转换；转置 W 的连续 32×64 tile 使用寄存器 shuffle，其他布局保留标量路径。
- **并行布局转换**：`T.copy(..., annotations={"hexagon.copy_workers": 6})`
  在普通 kernel 内同步分配互不重叠的布局 tile；同时允许调用线程预取下一个槽位。
  这是通用布局复制原语，不含 GEMM、RMSNorm 或预取策略。禁止在 `num_workers`
  kernel 中嵌套分发；只支持已匹配的完整 activation/output、转置 weight 布局。
- **GEMM 接口**：FP32 A `[M,K]`、FP16 W `[N,K]`、FP32 C `[M,N]`，计算 `A @ W.T`。
  内部使用 FP16 HMX，每次调用执行转换和打包，不要求预打包权重。K 整段累加，
  一个 HMX 单元，HVX workers 负责复制/布局转换。M/N/K/BN 为正的 32 倍数，
  N 可被 BN 整除，live set 必须放入实际 VTCM。

队列必须由同一 QuRT 线程提交和完成，不混用 raw/外部 DMA。并发 DDR 写须独占
缓存行，包括尾部周围的缓存行。上述证据来自 v79，不等于每 worker 独享物理 DMA
引擎。自动 `T.Pipelined`、任意嵌套调度、模型内集成尚未完成。

小块 source flush 的内联实现保留 `dccleana` 和 `syncht`：v79 PRM 的
“Data cache maintenance user operations”说明前者清理 L1D/L2，SDK 的
`qurt_memory.h` 明确 v60 及以上 cache-clean 调用执行 `syncht`。没有通过省略
一致性维护换取性能。

## 实现验证

2026-10-08 实现验证记录：完整 Hexagon 回归 **69 passed**；最终算子/归约/布局
设备回归 **84 条 PASS**。
独立 DMA worker 检查包括 120 次重复、18 次 64 KiB 块、18 次尾部、18 次 DSP 脏输入，
以及最后 3 次 HMX 错误恢复。

仓库内的 lowering 回归可用以下命令运行；上述真机专项验证脚本保留在仓库外的实验材料中。

```bash
source /home/lyn/workspace/hexagon-env.sh
export TILELANG_CACHE_DIR=/tmp/tl-worker-validation
export CMAKE_BUILD_PARALLEL_LEVEL=1
cd /home/lyn/workspace/tilelang-hexagon
cmake --build build --target tilelang -j2
python -m pytest -q testing/python/hexagon
```
