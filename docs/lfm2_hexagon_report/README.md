# LFM2-1.2B 在高通 Hexagon NPU 上的性能分析报告

> 本目录是一份**可汇报、可复现**的完整分析包:中文报告(本文件)+ 一键运行脚本
> ([`scripts/`](scripts))+ 可视化产物。面向"LFM2 端侧部署 / tilelang Hexagon 后端"的
> 性能评估。英文详版:[`../hexagon_lfm2_perf.md`](../hexagon_lfm2_perf.md)、
> [`../hexagon_profiling.md`](../hexagon_profiling.md)。

---

## 一、一句话结论

在骁龙 8 Elite(Hexagon v79)上,**llama.cpp 的 prefill 已接近 nexa(差 ~10%),但 decode
慢 2~2.6 倍**。这个差距**不是量化精度、也不是权重带宽,而是运行时/内核效率**:llama.cpp 的
decode 是逐 token 的 HVX 向量 GEMV,**把矩阵引擎 HMX 闲置了 95%**;nexa 走 QNN 整图编译能吃满
HMX + 更好的调度。**要追平 nexa,战场只有 decode 内核这一个。**

---

## 二、测试环境

| 项 | 值 |
|---|---|
| 设备 | OnePlus PJZ110 · **骁龙 8 Elite(SM8750**,平台 `sun`)· Android 15 |
| NPU | Hexagon **v79** —— 6 硬件线程、6× HVX(向量)、**1× HMX(矩阵)**、8 MB VTCM |
| 模型 | LFM2-1.2B —— 16 层(`n_embd` 2048,`n_ff` 8192,vocab 65536);**混合架构**:第 `[2,5,8,10,12,14]` 层是注意力,其余 10 层是门控短卷积(shortconv) |
| 运行时 | llama.cpp `4fc4ec55` + ggml-hexagon 后端(原版 skel `libggml-htp-v79.so`);Hexagon SDK 6.6.0.0 / Tools 19.0.07 / NDK r25c |
| 对照 | nexa / GenieX(= 高通收购后的 NexaSDK,QNN NPU 路径)—— 官方报告数字 |

---

## 三、性能对比(1024-token 输入)

`llama-bench -p 1024 -n 128 -r 3`(prefill 1024 token,再 decode 128,重复 3 次)。
复现:[`scripts/run_bench.sh`](scripts/run_bench.sh)。

| 运行时 | 权重 | 大小 | prefill (tok/s) | decode (tok/s) |
|---|---|---:|---:|---:|
| **nexa / GenieX**(官方报告,QNN) | **8-bit (w8a16)** | 1.27 GB | **3618.4** | **69.5** |
| llama.cpp | **Q8_0 (8-bit)** | 1.19 GB | 3271.6 | 26.6 |
| **llama.cpp** | **Q4_0 (4-bit)** | 0.65 GB | 3174.6 | 35.6 |
| llama.cpp | F16 (16-bit) | 2.18 GB | 2174.0 | 22.1 |

**三个关键发现:**

1. **llama.cpp 的 decode 受权重带宽限制** —— Q4_0(35.6)> Q8_0(26.6)> F16(22.1),
   位数越少越快,符合"逐 token GEMV 是访存瓶颈"的预期。**所以 Q4_0 是 llama.cpp 最快的 decode 配置。**

2. **同为 8-bit 时,nexa 的 decode 快 2.6 倍**(69.5 vs Q8_0 的 26.6)—— 比 Q4_0 对比(2 倍)
   差距更大。而且 **nexa 的 8-bit 甚至比 llama.cpp 的 4-bit 还快 2 倍**(69.5 vs 35.6),即便它
   要多搬一倍的权重字节。**⇒ nexa 的 decode 优势是纯运行时/内核的胜利**(QNN 整图编译 + 静态调度
   + 大概率用了 HMX,而 llama.cpp 在 decode 时把 HMX 闲置了 95%),与权重位数无关。

3. **prefill 差距小且与量化无关**(Q8_0 3272 ≈ Q4_0 3175,都约为 nexa 的 0.9×)—— prefill 是
   批量计算、走 HMX,这条路 llama.cpp 已经有竞争力。

> 说明:nexa 数字是官方公布值(我们**无法在设备上实跑 nexa** —— 其 license 校验失败);
> llama.cpp 数字是本次实测。

---

## 四、nexa `npu-mobile` 的权重类型:是 8-bit,不是 bfloat16

手机上 `NexaAI/LFM2-1.2B-npu-mobile` 的权重文件:`weights-1-2.nexa`(730 MB)+
`weights-2-2.nexa`(537 MB)= **1.27 GB / 1.17B 参数 = 1.08 字节/参数 = 8.7 bit/参数**。

- 这排除了 bfloat16(那样应该是 **2.34 GB**)。
- 是 **8-bit 权重(w8a16** —— 8-bit 权重、16-bit 激活,QNN LLM 导出的标准配置),多出来的
  ~0.7 bit 是逐通道 scale + 可能的 fp16 embedding/norm。
- `.nexa` 是加密/压缩的私有容器(只能读出 magic `NEXAG` 和 QNN 张量名如 `past_conv_8_out`),
  无法在设备上抽出精确 QNN datatype,但**字节预算是铁证:不是 bf16,是 ~8-bit**。

这也让上面的 **Q8_0(8.5 bit/参数)对比成为最干净的同精度对照**。

---

## 五、算子映射:哪些跑在 NPU、哪些在 HMX / HVX / scalar

用 ggml-hexagon 内置 profiler(`GGML_HEXAGON_PROFILE=2` —— 每算子记录 usec/cycles + 8 个硬件
PMU 计数器)采一次 prefill + 32-token decode:**7994 个算子实例,合计 500.3 ms HTP 计算时间**。
复现:[`scripts/run_profile.sh`](scripts/run_profile.sh)、[`scripts/run_op_split.sh`](scripts/run_op_split.sh)。

### 5.1 NPU vs CPU

**整个 transformer 都在 NPU(HTP)上跑。** 在 HTP 执行的算子:`MUL_MAT`(及融合变体
`MUL_MAT+MUL_MAT`、`MUL_MAT+ADD`、`MUL_MAT+MUL_MAT+MUL_MAT`)、`FLASH_ATTN_EXT`、
`RMS_NORM+MUL`、`SSM_CONV`、`SWIGLU`、`CONCAT`、`CPY`、`ROPE`、`SET_ROWS`、`MUL`、`ADD`、
`SCALE`、`GET_ROWS`。`PERMUTE`/`RESHAPE`/`VIEW`/`NONE` 是**零成本布局操作**(不计算)。只有最后的
采样在 CPU。

> **注意:ggml-hexagon 本身已经在做算子融合**(`OPFUSION=1`):`ffn_gate+ffn_up`、
> `ffn_down+残差add`、QKV 三个投影、`rms_norm+权重mul` 各自都被融合成一个 HTP 算子。

### 5.2 HMX vs HVX vs scalar —— 用硬件 PMU 计数器验证

依据 `kparams` 内核标签 **和** 片上 `HVX_ACTIVE` PMU 计数器(事件 `0x100`;比值 = HVX活跃周期/总周期,
因跨 6 线程 worker-pool 累加可 >1):

| 引擎 | 占比 | 跑什么 |
|---|---:|---|
| **HMX(矩阵)** | **5.2 %** | **只有** prefill 的批量 matmul(`hmx-tiled`)+ 注意力(`hmx-pipe`) |
| **HVX(向量)** | **~95 %** | 其余全部(**decode 的所有 GEMV** + flash-attn + 逐元素/norm/conv/swiglu) |
| **scalar** | **≈0 %** | 没有算术跑标量单元 |

**"哪些跑在 scalar 上"的答案:基本没有。** 唯一非 HVX/非 HMX 的只是**内存搬运**:`GET_ROWS`
(embedding gather)和 `SET_ROWS`(KV/conv-cache scatter)是标量线程驱动的访存,HVX_ACTIVE≈0。
那几个小的 `MUL`/`ADD`/`SCALE`/`ROPE` **也发 HVX 向量指令**(源码用 `Q6_V*` 内联),只是数据太少、
被固定的**每算子派发开销**盖过(overhead-bound)—— 而融合正是消掉这块开销的手段。

### 5.3 时间都花在哪(500.3 ms)

| 模块 | 占比 | 明细 |
|---|---:|---|
| **FFN** | **~66 %** | `ffn_gate+ffn_up`(融合)41.8% + `ffn_down+add`(融合)22.5% + `SWIGLU` 1.6% |
| **短卷积** | **~15 %** | `in_proj` 10.1% + `out_proj` 3.5% + `SSM_CONV` 1.1% |
| **注意力** | **~11 %** | `FLASH_ATTN` 5.4% + QKV 3.0% + `attn_output` 2.2% |

**FFN 占了三分之二,而且 ggml 已经把它融合好了**(gate+up、down+add)。**短卷积块没被融合**
(`in_proj → SSM_CONV → out_proj`,因为 conv 夹在两个投影中间)—— 这是留给 tilelang 的融合空档。

---

## 六、结论与对 tilelang 的意义

1. **matmul = ~85% 的计算,而且是 HVX GEMV,不是 HMX。** decode 全是 `ne1=1` 的向量 GEMV,
   受访存带宽限制;HMX 矩阵引擎 95% 闲置。**杠杆 = decode 内核(HVX GEMV + 4-bit 反量化),或把 decode 搬上 HMX。**
2. **FFN(66%)ggml 已经融合** —— 再融收益有限,值钱的是 matmul 内核本身。
3. **短卷积块(15%)没融合 —— 这才是 tilelang 该做的子图融合目标**(中间结果留在 VTCM,省两次 DDR 往返)。
4. **单算子 matmul 已追平;到 nexa 的 2~2.6× decode 差距是运行时/调度差距**,不是算子覆盖问题。

---

## 七、如何复现

前提:`adb` 已连上手机;设备 `/data/local/tmp/llamahtp/` 下已部署 llama.cpp(见
[`scripts/env.sh`](scripts/env.sh) 顶部清单)。全部脚本从设备读写,主机只需 adb。

```bash
cd docs/lfm2_hexagon_report/scripts

./make_q8.sh          # 从 F16 生成 Q8_0(8-bit,对齐 nexa 精度;纯 CPU 量化)
./run_bench.sh        # 性能:Q4_0/Q8_0/F16 @ 1024 prefill + 128 decode(第三节的数)
./run_op_split.sh     # 算子的 NPU/CPU 归属(第 5.1 节)
./run_profile.sh      # 逐算子 PMU 剖析 + 生成 timeline/perfetto(第 5.2/5.3 节)
```

**三个踩过的坑(脚本已处理)**:
1. `profile-op` 日志走 `GGML_LOG_DEBUG`,必须同时开 `GGML_HEXAGON_VERBOSE=1` **和** `--verbose`;
2. 日志是 non-ISO ASCII,解析要 `grep -a` / latin1;
3. **算子名可能是融合的(`MUL_MAT+MUL_MAT`),正则要用 `[\w+]+` 而非 `\w+`** —— 否则会静默漏掉
   ~40% 的算子并得出完全相反的结论(这是本报告修正过的一个真实错误)。

---

## 八、文件清单

```
docs/lfm2_hexagon_report/
├── README.md                     本报告(中文)
├── scripts/
│   ├── env.sh                    公共环境 + adb 包装
│   ├── make_q8.sh                F16 → Q8_0 量化
│   ├── run_bench.sh              性能测试(4 配置 @1024)
│   ├── run_op_split.sh           算子 NPU/CPU 归属
│   ├── run_profile.sh            逐算子 PMU 剖析 + 可视化
│   ├── profile_lfm2.py           逐算子汇总表(按类型/权重/引擎)
│   ├── timeline_export.py        nsys 风格 timeline 数据导出
│   └── chrome_trace_export.py    Perfetto/Chrome-trace JSON 导出
└── results/                      运行脚本后生成(prof_pmu.log / *.json)
```

**可视化产物**(在上级 `docs/`):
- [`../hexagon_timeline.html`](../hexagon_timeline.html) —— 交互式 nsys 风格时间线,7994 个算子按引擎分道、可缩放悬停。
- [`../lfm2_perfetto_trace.json`](../lfm2_perfetto_trace.json) —— 拖进 [ui.perfetto.dev](https://ui.perfetto.dev):3 条引擎轨道 + HVX_ACTIVE/committed-pkt/AXI 硬件计数器折线。

**英文详版**:[`../hexagon_lfm2_perf.md`](../hexagon_lfm2_perf.md)(完整报告)、
[`../hexagon_profiling.md`](../hexagon_profiling.md)(profiling 方法学 + itrace 尝试 + PMU 解码)。
