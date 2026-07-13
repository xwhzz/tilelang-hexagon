#!/usr/bin/env bash
# 逐算子性能剖析:用 ggml-hexagon 内置 profiler 采每个算子的 usec/cycles + 8 个硬件 PMU 计数器,
# 然后跑分析脚本,产出:逐算子表 + nsys 风格 timeline JSON + Perfetto trace JSON。
#
# 用法: ./run_profile.sh [模型 Q4_0|Q8_0|F16] [decode token 数]   默认 Q4_0 32
#
# 关键点(踩过的坑):
#   1) profile-op 日志走 GGML_LOG_DEBUG,必须同时开 GGML_HEXAGON_VERBOSE=1 和 --verbose
#   2) 日志是 non-ISO ASCII,解析要用 grep -a / latin1
#   3) 算子名可能是融合的("MUL_MAT+MUL_MAT"),正则要用 [\w+]+ 而不是 \w+,否则漏掉 ~40% 算子
DIR="$(cd "$(dirname "$0")" && pwd)"; source "$DIR/env.sh"
M="${1:-Q4_0}"; N="${2:-32}"
OUT="$DIR/../results"; mkdir -p "$OUT"
PROMPT="The history of artificial intelligence spans several decades of research beginning in the nineteen fifties when pioneers first explored whether machines could simulate human reasoning"

hexkill
echo "== 采集 GGML_HEXAGON_PROFILE=2 (usec + cycles + 8 PMU 计数器) | 模型 $M | decode $N =="
adb shell "cd $DEV && LD_LIBRARY_PATH=. $ADSP \
  GGML_HEXAGON_PROFILE=2 GGML_HEXAGON_VERBOSE=1 \
  ./llama-cli -m LFM2-1.2B-$M.gguf --device HTP0 -ngl 99 -n $N -st --verbose -p '$PROMPT' 2>&1" \
  2>/dev/null | tr -d '\r' > "$OUT/prof_pmu.log"
hexkill

N_OPS=$(grep -ac profile-op "$OUT/prof_pmu.log")
echo "采到 profile-op 行数: $N_OPS"
[ "$N_OPS" -eq 0 ] && { echo "无数据 —— 检查 GGML_HEXAGON_VERBOSE / --verbose"; exit 1; }

echo ""; echo "======== 逐算子汇总表 ========"
python3 "$DIR/profile_lfm2.py"       "$OUT/prof_pmu.log"
echo ""; echo "======== 生成可视化数据 ========"
python3 "$DIR/timeline_export.py"    "$OUT/prof_pmu.log" "$OUT/timeline.json"
python3 "$DIR/chrome_trace_export.py" "$OUT/prof_pmu.log" "$OUT/lfm2_perfetto_trace.json"
echo ""
echo "产物在 $OUT/ :"
echo "  prof_pmu.log             原始 profile 日志"
echo "  timeline.json            nsys 风格 timeline 数据(供 hexagon_timeline.html)"
echo "  lfm2_perfetto_trace.json 拖进 https://ui.perfetto.dev 打开"
