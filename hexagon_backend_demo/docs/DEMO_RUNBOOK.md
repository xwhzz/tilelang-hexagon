# 现场演示手册

## 1. 演示定位

主线是 TileLang Hexagon backend，不是某个模型。推荐顺序：

1. 展示硬件与五层 backend 架构；
2. lower 一个普通 `T.gemm`，查看生成的 VTCM/HMX C；
3. 在设备运行普通 matmul 和 worker pool；
4. 用 Q4 说明低层 layout/intrinsic/DMA schedule；
5. 用 llama.cpp 说明如何把 generated body 嵌入宿主 runtime；
6. 用 Q8 说明同一 backend 也覆盖 HVX decode，而不是只服务 HMX/Q4；
7. 最后讲项目评估与下一阶段。

## 2. 演示前检查

```bash
cd /home/xwh/tilelang-hexagon/hexagon_backend_demo
./scripts/demo.sh check
./scripts/demo.sh status
```

确认设备当前 active skel 与 `.stock` 哈希相同。关闭可能占用 HTP session 的其他 llama 进程。

## 3. 12 分钟完整路线

### A. 通用 backend lower（2 分钟）

```bash
./scripts/demo.sh lower
```

输出应包含：

- `tl_vtcm_base`：`T.alloc_shared` 已映射到 VTCM；
- block loops：GPU grid 已转为 DSP 控制流；
- `tl_hexagon_hmx_gemm`：`T.gemm` 已选择 HMX recipe。

生成的完整 C 保存在 `results/generated_matmul_hexagon.c`。

### B. 离线 backend contract（1 分钟）

```bash
./scripts/demo.sh test-codegen
```

无需设备，验证 HMX layouts、显式 atoms、Q4 staged manifests 和 Q8 dot lowering。

### C. Standalone 设备能力（3 分钟）

```bash
./scripts/demo.sh matmul
./scripts/demo.sh workers
```

第一条展示通用 TileLang `T.gemm` 在 HMX 正确执行；第二条展示 1-HMX/6-HVX worker mapping。

### D. 低层量化案例（2 分钟）

```bash
./scripts/demo.sh q4-standalone
./scripts/demo.sh q8-standalone
```

Q4 展示 layout + native dequant tile + explicit HMX atoms；Q8 展示 HVX signed-int8 dot。可以只跑一条，
另一条在演示站中说明。

### E. 框架嵌入和真实模型（3 分钟）

```bash
./scripts/demo.sh llama-walkthrough
./scripts/demo.sh inspect-llama
./scripts/demo.sh deploy-q4
./scripts/demo.sh bench-q4
./scripts/demo.sh chat-q4
./scripts/demo.sh restore-stock
```

这里的目的不是证明 backend 只支持 LFM2，而是证明 TileLang generated body 能使用宿主提供的资源，
进入真实模型 forward 并保持 fallback/correctness。

`llama-walkthrough` 实际生成一个 `M=32,N=256,K=2048` kernel，并逐步展示从 PrimFunc 到 patch；
`inspect-llama` 依次显示完整集成的 generated entry symbols、HTP CMake source list、stock dispatch seam、
adapter 的 `matches/run` 与注册点。详细源码 walkthrough 见 `docs/LLAMA_CPP_INTEGRATION.md`。

### F. 项目评估（1 分钟）

回到演示站的“任务评估”页：当前是 device-validated research backend；优先补 async DMA target hook、
runtime context、测试矩阵和 autotune。

## 4. 5 分钟精简路线

```bash
./scripts/demo.sh lower
./scripts/demo.sh matmul
./scripts/demo.sh deploy-q4
./scripts/demo.sh chat-q4
./scripts/demo.sh restore-stock
```

只讲演示站中的“硬件映射”“编译流程”“两种运行模式”“Q4 案例”“任务评估”五部分。

## 5. 无设备备用路线

```bash
./scripts/demo.sh lower
./scripts/demo.sh test-codegen
./scripts/demo.sh generate-q4
```

然后展示：

- `results/generated_matmul_hexagon.c`；
- `results/generated/kernel_qmatmul_hmx_staged_*.cc`；
- `sources/tilelang/examples/hexagon/llama_cpp_integration/*.manifest.json`；
- `results/raw/q4-staged-generate.txt`；
- 演示站的性能与正确性页面。

## 6. 常见故障

| 现象 | 处理 |
|---|---|
| FastRPC `0x80000406` | 确认命令在 `DEVICE_DIR` 执行，ADSP path 以 `.` 开头 |
| VTCM/session 失败 | 结束残留 llama/agent 进程，确认没有并发 HTP session |
| 改 skel 后仍像旧版本 | 结束旧进程后重新部署，检查 `demo.sh status` 哈希 |
| `pytest` 不存在 | 使用 `demo.sh test-codegen`，它不依赖 pytest |
| 模型演示中断 | 先 `restore-stock`，再检查 active/stock 哈希 |

## 7. 演示结束

```bash
./scripts/demo.sh restore-stock
./scripts/demo.sh status
```

必须看到 active skel 和 stock skel 的 SHA256 相同。
