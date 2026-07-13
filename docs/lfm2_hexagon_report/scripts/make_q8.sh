#!/usr/bin/env bash
# 从 F16 GGUF 生成 Q8_0(8-bit,与 nexa npu-mobile 的权重精度一致)。
# 量化是纯 CPU 操作,在手机上直接跑,输入是设备上已有的 F16 GGUF。
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"; source "$DIR/env.sh"

echo "== 量化 LFM2-1.2B-F16.gguf -> LFM2-1.2B-Q8_0.gguf (Q8_0) =="
adb shell "cd $DEV && LD_LIBRARY_PATH=. \
  ./llama-quantize LFM2-1.2B-F16.gguf LFM2-1.2B-Q8_0.gguf Q8_0" 2>/dev/null | tr -d '\r' | tail -5

echo "== 产物 =="
adb shell "ls -la $DEV/LFM2-1.2B-Q8_0.gguf" 2>/dev/null | tr -d '\r' | awk '{print $5, $NF}'
