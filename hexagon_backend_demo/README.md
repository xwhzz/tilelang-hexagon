# TileLang Qualcomm Hexagon Backend Demo

这是 TileLang 适配 Qualcomm Hexagon NPU 的集中交付目录。主目标是说明并演示一个完整 backend
如何从 TileLang DSL 走到 Hexagon HVX/HMX/VTCM，再以 standalone FastRPC 或嵌入宿主 runtime
两种方式执行。Q4、Q8 和 llama.cpp 是端到端案例，不是 backend 的边界。

## 演示入口

- 浏览器演示站：[`index.html`](index.html)
- 16:9 PDF 演示稿：[`hexagon_backend_technical_route.pdf`](hexagon_backend_technical_route.pdf)
- 逐页讲稿：[`docs/SPEAKER_NOTES.md`](docs/SPEAKER_NOTES.md)
- 现场演示手册：[`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md)
- 主技术路线：[`docs/TECHNICAL_ROUTE.md`](docs/TECHNICAL_ROUTE.md)
- Backend 架构与抽象参考：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- 生成代码接入 llama.cpp：[`docs/LLAMA_CPP_INTEGRATION.md`](docs/LLAMA_CPP_INTEGRATION.md)
- 单 shape 分步例子：[`walkthrough/llama_cpp_q4/README.md`](walkthrough/llama_cpp_q4/README.md)
- 项目评估与任务列表：[`docs/PROJECT_EVALUATION.md`](docs/PROJECT_EVALUATION.md)
- 实测结果摘要：[`results/RESULTS.md`](results/RESULTS.md)

`index.html` 可以直接双击打开，也可以启动本地静态服务：

```bash
./scripts/demo.sh serve
# open http://127.0.0.1:8765
```

需要在修改站点后重新生成 PDF 时，先安装 `requirements-presentation.txt`，再运行：

```bash
./scripts/render_pdf.sh
```

## 最短演示流程

```bash
./scripts/demo.sh check
./scripts/demo.sh lower
./scripts/demo.sh test-codegen
./scripts/demo.sh matmul
./scripts/demo.sh workers
```

以上五步已经覆盖 target 注册、TIR lowering、Hexagon C codegen、HMX `T.gemm`、VTCM、FastRPC
部署和 1-HMX/6-HVX worker pool。随后可以选择 Q4 或 Q8 案例：

```bash
./scripts/demo.sh q4-standalone
./scripts/demo.sh q8-standalone
```

最后用 llama.cpp 展示生成算子如何嵌入真实框架：

```bash
./scripts/demo.sh llama-walkthrough
./scripts/demo.sh inspect-llama
./scripts/demo.sh deploy-q4
./scripts/demo.sh bench-q4
./scripts/demo.sh chat-q4
./scripts/demo.sh restore-stock
```

## 目录

```text
hexagon_backend_demo/
├── index.html                    离线技术演示站
├── hexagon_backend_technical_route.pdf
│                                 可直接投屏的 16:9 PDF
├── assets/                       演示站静态资源
├── docs/                         技术路线、讲稿、runbook、项目评估
├── scripts/                      lower、测试、设备和模型演示入口
├── walkthrough/llama_cpp_q4/     单 shape 的 kernel -> generator -> adapter -> patch
├── sources/
│   ├── tilelang/                 本次 backend 相关源码快照
│   ├── llama.cpp/                宿主集成点的源码快照
│   └── SOURCE_SHA256SUMS.txt     快照校验
├── artifacts/                    已验证的 v79 stock/Q4/Q8 DSP skel
└── results/                      性能摘要、生成源码和原始日志
```

`sources/` 是便于讲解和归档的快照，开发时仍以主仓库为准。更新快照：

```bash
./scripts/refresh_sources.sh
```

## 演示环境

当前机器的可运行配置在 `config.sh`，可移植模板在 `config.example.sh`。关键依赖：

- TileLang 本仓库及其 `tl` Python 环境；
- Hexagon SDK 6.6、Hexagon Tools 19.0.07、Android NDK；
- 已授权的 Qualcomm Hexagon v79 设备；
- llama.cpp `4fc4ec55` 及其 Snapdragon build；
- 设备目录中已有 `llama-cli`、`llama-bench` 和模型。

FastRPC DSP loader 对工作目录敏感。模型命令必须在设备的 `DEVICE_DIR` 中执行，并使用
`LD_LIBRARY_PATH=.` 和 `ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp`；`demo.sh` 已固定该规则。

## 安全约定

- `deploy-q4` 会替换设备当前活动的 DSP skel，但不会覆盖 `.stock` 备份；
- 演示结束始终运行 `restore-stock`；
- 未匹配的 dtype、shape、bias 或失败路径必须返回 `-1`，让宿主执行 stock kernel；
- `artifacts/SHA256SUMS` 记录本次验证产物的哈希。
