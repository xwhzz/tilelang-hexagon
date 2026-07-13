#!/usr/bin/env bash
# 性能测试:1024-token prefill(pp1024)+ 128-token decode(tg128),3 次重复取均值±标准差。
# 对 Q4_0 / Q8_0 / F16 三种权重各跑一遍,与 nexa(QNN,8-bit)对比。
#
# 用法: ./run_bench.sh [模型列表]   默认 "Q4_0 Q8_0 F16"
DIR="$(cd "$(dirname "$0")" && pwd)"; source "$DIR/env.sh"
MODELS="${*:-Q4_0 Q8_0 F16}"

hexkill
for M in $MODELS; do
  echo "==================== LFM2-1.2B-$M ===================="
  hexrun "./llama-bench -m LFM2-1.2B-$M.gguf -dev HTP0 -ngl 99 -p 1024 -n 128 -r 3"
  echo ""
done
hexkill
# 参考结果(Snapdragon 8 Elite / Hexagon v79):
#   Q4_0(4-bit): pp1024 ~3175 / tg128 ~35.6
#   Q8_0(8-bit): pp1024 ~3272 / tg128 ~26.6
#   F16(16-bit): pp1024 ~2174 / tg128 ~22.1
#   nexa(8-bit,官方报告): prefill 3618 / decode 69.5
