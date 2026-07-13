#!/usr/bin/env bash
# 判定哪些算子跑在 NPU(HTP)上、哪些回退 CPU:抓 ggml-hexagon 的 supports-op 判定
# (yes = HTP 接受,no = CPU 回退)与实际 execute-op。
DIR="$(cd "$(dirname "$0")" && pwd)"; source "$DIR/env.sh"
M="${1:-Q4_0}"
OUT="$DIR/../results"; mkdir -p "$OUT"

hexkill
adb shell "cd $DEV && LD_LIBRARY_PATH=. $ADSP GGML_HEXAGON_VERBOSE=1 \
  ./llama-cli -m LFM2-1.2B-$M.gguf --device HTP0 -ngl 99 -n 2 -st --verbose -p Hi 2>&1" \
  2>/dev/null | tr -d '\r' > "$OUT/supports.log"
hexkill

python3 - "$OUT/supports.log" <<'PY'
import re, sys
from collections import defaultdict
sup=defaultdict(lambda:{'yes':0,'no':0}); ex=defaultdict(int)
for l in open(sys.argv[1], encoding='latin1'):
    m=re.search(r'supports-op ([\w+]+)\|.*\|(yes|no)$', l.rstrip())
    if m: sup[m.group(1)][m.group(2)]+=1
    e=re.search(r'execute-op ([\w+]+)\|', l)
    if e: ex[e.group(1)]+=1
print(f"{'op':<18}{'NPU-ok':>7}{'reject':>7}{'HTP-exec':>9}   判定")
for op in sorted(set(sup)|set(ex)):
    y,n,e=sup[op]['yes'],sup[op]['no'],ex[op]
    v="NPU执行" if e>0 else ("零成本布局/融合" if y>0 else "CPU")
    print(f"{op:<18}{y:>7}{n:>7}{e:>9}   {v}")
PY
