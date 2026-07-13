# 原始 log 与核对说明

本目录是报告([`../README.md`](../README.md))所有数字的**原始证据**,每个文件对应报告的哪一节、
怎么复核都列在下面。设备:OnePlus PJZ110 / 骁龙 8 Elite(SM8750)/ Hexagon v79。

| 文件 | 对应报告 | 内容 | 怎么核对 |
|---|---|---|---|
| `bench_raw.log` | §三 性能 | llama-bench 完整原始输出(Q4_0/Q8_0/F16,含 banner+表+build) | `grep -E 'pp1024|tg128' bench_raw.log` |
| `prof_pmu.log.gz` | §五 算子 | 7994 个算子的逐条 PMU 采集(usec/cycles+8 计数器) | `zcat prof_pmu.log.gz | grep -a 'profile-op MUL_MAT+MUL_MAT' | head` |
| `analysis_perop.txt` | §5.3 时间分布 | `profile_lfm2.py` 汇总:逐算子表 + 按权重角色 + 引擎(HMX/HVX)划分 | 直接看 |
| `op_support_raw.log` | §5.1 NPU/CPU | ggml-hexagon 的 supports-op / execute-op 原始判定(14895 行) | `grep -a 'supports-op RMS_NORM' op_support_raw.log | head` |
| `op_split_summary.txt` | §5.1 NPU/CPU | 上面的汇总:每类算子 NPU-ok / reject / 实际执行 | 直接看 |
| `weight_type.log` | §四 权重类型 | nexa npu-mobile `.nexa` 文件大小 + 字节/参数换算 | 直接看 |

## 关键数字速查(从这些 log 得出)

**性能 @1024**(`bench_raw.log`,本次采集,与报告有 ±2% run-to-run 抖动属正常):

```
Q4_0(4-bit): pp1024 3200.32 / tg128 36.95
Q8_0(8-bit): pp1024 3191.66 / tg128 26.97
F16 (16-bit): pp1024 2173.84 / tg128 22.02
nexa(8-bit,官方报告): 3618 / 69.5      <- 无法本地实跑(license 校验失败)
```

**权重类型**(`weight_type.log`):1.27 GB / 1.17B = **8.7 bit/参数 → 8-bit,非 bf16**(bf16 应 2.34 GB)。

**算子/引擎**(`analysis_perop.txt`):7994 算子 / 500.3 ms;matmul(含融合)~85%;
FFN 融合(`MUL_MAT+MUL_MAT`=ffn_gate+up 41.8%,`MUL_MAT+ADD`=ffn_down 22.5%)主导;
HMX 只 5.2%,HVX ~95%。

## 重新生成

```bash
cd ../scripts
./run_bench.sh        # -> 覆盖 ../results/*(bench 部分需自己重定向)
./run_op_split.sh     # -> op_support_raw.log / op_split_summary.txt
./run_profile.sh      # -> prof_pmu.log + timeline.json + lfm2_perfetto_trace.json
```

> 注:`run_profile.sh` 会在本目录生成**未压缩**的 `prof_pmu.log`;这里提交的是压缩版
> 以控制仓库体积。`analysis_perop.txt` = `python3 ../scripts/profile_lfm2.py <(zcat prof_pmu.log.gz)`。
