# 端侧 tilelang 算子接入为什么比 CUDA 难

一句话:**CUDA 是「按引用替换一个 Python callable」,端侧是「按结构身份、在部署期、把一个算子塞进别人的、跑在另一块协处理器上的、还不一定给你完整硬件权限的 runtime 里」。** 下面把每条差异和它的根源列清楚——这些是我们这一轮实测踩下来的,不是空想。

| # | CUDA / server | 端侧(Hexagon) | 难在哪 |
|---|---|---|---|
| 1 | 运行时就是 **Python 解释器**,模型是 Python 程序 | 设备上**没有 Python**,模型被冻结成图 IR(ONNX)/ 手写 C++(llama.cpp)/ 编译 blob(QNN) | 不能「按对象引用换掉一个 callable」;只能**按结构身份(op+dtype+shape 谓词)在加载/切分期匹配**。绑定从作者期挪到了部署期。 |
| 2 | PyTorch eager 直接调你的 callable | 你插进的是**别人的 runtime**(llama.cpp/ONNXRuntime/QNN/ExecuTorch),它掌管派发、内存、调度 | 每个 runtime 扩展点(backend / delegate / custom-op)都不一样;kernel 可移植,但**胶水按 runtime 一份一份写**。 |
| 3 | 任何 kernel 都能用 GPU | **扩展点决定你能不能拿到 HMX**:自研 backend / 自己的 PD 能拿到;QNN 的 external op-package **拿不到(实测三层焊死)** | 硬件访问被扩展点门控。这条直接把架构从「换单个 op」逼成「**在你自己拥有的保护域(PD)里拥有一段子图**」。 |
| 4 | 你拿到的是一个连续的 torch tensor,同一块显存 | 张量是**量化的(q4_0)+ repack 过的(HMX tile 版式)+ 住在 rpcmem/ION 与协处理器共享** | kernel 必须吃/吐这套确切格式、共享内存零拷贝,而不是接一个 torch tensor。反量化的字节布局、repack 的 tile 排法都得对齐到 bit。 |
| 5 | PyTorch 按 shape 运行期派发 + CUDA JIT,动态形状被隐藏 | tilelang codegen 出的是**定长 kernel**,而模型形状是变的(prefill M=seq / decode M=1;每个权重 K,N 不同) | 要么按 shape 一族一族生成,要么在 adapter 里手工 tiling/padding。而且 decode 的 M=1 是 **GEMV(访存瓶颈)**,拿 HMX 的定长 GEMM 去 pad 到 32 是 32× 浪费——形状语义直接影响该用哪种 kernel。 |
| 6 | 一个 `.so`/callable 加载进进程就行 | kernel 必须**编译进 host runtime 的 DSP skel(同一个 PD / 同一个 TU)**,因为要共用那个 skel 的 VTCM/HMX/worker-pool;独立的 FastRPC skel **嵌不进去** | 需要「可嵌入 kernel」产物(一个 C 函数 + 运行时头),而不是 `tilelang.compile` 出的独立 skel。我们现在的运行时用 `static` 全局态,还被逼成「一个算子一个 TU」——就是这条的代价。 |
| 7 | kernel 在同进程、同一块 GPU 上跑,launch 就是一次调用 | kernel 跑在**另一块协处理器(cDSP)、另一个保护域**,靠 **FastRPC 异步命令队列 + 共享内存**过去 | 张量 marshal、内存常驻、session 生命周期、跨 PD 的 HMX 锁/VTCM 归属,全变成你要管的东西(我们的 bridge 就是在解决这个:蹭 host 的 VTCM+HMX,不自己申请)。 |
| 8 | 融合 = 写一个融合 kernel,你拥有整个 Python,随便融 | 融合的收益(省掉中间张量的 DDR 往返)**要求你拥有一段多算子子图**;而多数 op 级扩展点(QNN op-package)是单 op 且被墙 | 「想做融合算子」在数学上**强制**你走「自己 PD + 拥有子图」这条路。这也是 tilelang 少数能**赢过**而非打平调优后端的方向。 |
| 9 | Python REPL、print、秒级迭代 | 交叉编译 aarch64 + cDSP、SDK 版本要对齐、FARF 默认到不了 logcat、rebuild→push→run 一轮几分钟、崩溃是 DSP 的 SIGABRT | 迭代摩擦是 CUDA 的百倍量级;设备正确性只能靠「协程输出连贯」或 abort/返回码当信号。 |
| 10 | 你控制精度(fp16/bf16) | 模型**已离线量化**(q4_0/q6_K),你的 kernel 必须复刻反量化数学 + 后端的 repack;有些 dtype(q6_K)后端根本不做 → 落 CPU | 量化把「作者控制精度」变成「必须适配既定量化格式」;而且量化选择(哪层留 q6_K)反过来决定了哪些算子在 NPU 上。 |

## 三个最本质的

如果只记三条:

1. **绑定方式变了**(第 1 条):CUDA 按引用换 callable;端侧按结构身份在部署期匹配。所以我们需要「kernel + manifest 谓词」而不是「一个 Python 对象」。
2. **硬件访问被 PD 门控**(第 3、7、8 条):你的 HMX 算子只能跑在**你自己拥有 HMX 的保护域**里。QNN 的教训不是「QNN 不行」,而是「**任何沙箱化、不给你 HMX 的扩展点都不行**」。这把边界从单 op 逼到「你拥有的子图/backend」,也顺带打开了融合的门。
3. **kernel 不再自足**(第 5、6 条):定长 codegen + 必须可嵌入到别人的 skel。这两条是我们把 PoC 变成干净架构最需要补的工程(可嵌入 kernel target + 显式 `tl_ctx`)。

## 对策(和这些难点一一对应)

- 对 1 → **kernel 注册表**,按 op+dtype+shape 谓词匹配;
- 对 2 → **每个 runtime 一个薄 adapter**(llama.cpp 的 backend hook / 未来 ExecuTorch delegate);
- 对 3/7/8 → **永远在你自己的 PD 里调 kernel**(bridge 蹭 host 的 VTCM+HMX),融合走「拥有子图」;
- 对 5/6 → tilelang 出 **可嵌入 kernel 产物** + 把运行时 `static` 全局改成显式 context;
- 对 4/10 → adapter 负责对齐 repack 版式与反量化数学;
- 对 9 → 把可复现的 build/run/trace 命令固化下来(见 `docs/llama_cpp_reproduce.md`)。

详见 `docs/llama_cpp_integration.md`(嵌入机制)、`docs/llama_cpp_inference_flow.md`(整条推理链)、
`docs/executorch_delegate_evaluation.md`(通用平台方向,若已生成)。
