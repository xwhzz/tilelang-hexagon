# Device Validation Summary

设备：Qualcomm Hexagon v79，6 HVX contexts，1 HMX，8 MiB VTCM。

## 交付复验（2026-07-16）

| 演示步骤 | 本次结果 |
|---|---|
| offline generic lower | 生成 C 包含 VTCM high-water、`half64` copy 和 `tl_hexagon_hmx_gemm` |
| offline Q4/Q8 contracts | `7/7` PASS |
| FP16 HMX `256x256x256` | max abs `0.001222`，PASS |
| 6-worker HVX-heavy grid | `1058 ms -> 266 ms`，`3.97x` |
| Q4/HMX `32x128x512` | relative error `4.409e-4`，PASS |
| Q8/HVX `32x2048` | max abs `7.62939e-6`，PASS |
| llama.cpp Q4 pp32 | `652.22 +/- 50.34 tok/s`，3 runs |
| llama.cpp deterministic generation | `The capital of France is Paris. Paris` |
| 演示后设备状态 | active skel 与 stock SHA256 均为 `4417387b...e6a8957b05` |

## 通用 backend

| 能力 | 结果 |
|---|---|
| FP16 `T.gemm -> HMX` | 数值通过；现有 standalone benchmark 约 17 TFLOPS |
| `T.Kernel(num_workers=6)` | HVX-heavy 示例多次复验约 3.9x |
| Flash attention / RMSNorm | 真实设备 golden 验证通过 |
| FastRPC persistent agent | build/deploy/run 路径已打通 |

## Q4/HMX 案例

| 验证 | 结果 |
|---|---|
| `32x128x512` standalone | relative error `4.4e-4`，PASS |
| `32x512x512` standalone | relative error `4.0e-4`，PASS |
| LFM2-1.2B Q4 pp32，serial N=512 baseline | `127.39 tok/s` |
| staged TileLang，5 runs | `616.78 +/- 29.77 tok/s` |
| stock，5 runs | `604.36 +/- 36.28 tok/s` |
| final staged build，3 runs | `643.00 +/- 6.53 tok/s` |
| 2048x8192 operator | staged `0.375-0.392 ms`；stock `0.385-0.408 ms` |

两组重复数据方差重叠，因此结论是进入 stock 性能档位，不宣称稳定加速。

固定 prompt、temperature 0、seed 123 的输出与 stock 一致：

```text
The capital of France is Paris. Paris
```

## Q8/HVX 案例

| 验证 | 结果 |
|---|---|
| Q8_0 `32x2048` atom | max abs `7.62939e-6`，PASS |
| Q8_0 `32x8192` atom | max abs `1.90735e-5`，PASS |
| TileLang Q8 model pp1024 | `3300.53 +/- 63.94 tok/s` |
| stock Q8 model pp1024 | `3331.28 +/- 20.96 tok/s` |
| TileLang Q8 model tg128 | `35.67 +/- 0.37 tok/s` |
| stock Q8 model tg128 | `35.46 +/- 0.17 tok/s` |

Q8 案例验证了 generated HVX dot atom 的模型集成与性能 parity；宿主原有 DMA、VTCM 和 worker
流水线保持不变。

## 证据文件

- `raw/profile-q4-n512.txt`：串行 Q4 profile；
- `raw/profile-q4-staged.txt`：staged Q4 profile；
- `raw/profile-q4-stock.txt`：stock profile；
- `raw/q4-staged-generate.txt` 和 `raw/q4-stock-generate.txt`：确定性模型输出。
