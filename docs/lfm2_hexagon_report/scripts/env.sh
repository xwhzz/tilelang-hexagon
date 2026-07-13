#!/usr/bin/env bash
# 公共环境:设备部署目录 + 在设备上带正确库路径运行命令的包装函数。
# 其它脚本都 source 本文件。
#
# 前提:
#   - 本机已装 adb,且手机已通过 `adb devices` 授权连接
#   - 设备 /data/local/tmp/llamahtp/ 下已部署:
#       llama-cli / llama-bench / llama-quantize
#       libggml*.so / libllama*.so / libc++_shared.so
#       libggml-htp-v79.so           (Hexagon DSP skel)
#       LFM2-1.2B-{Q4_0,F16}.gguf     (Q8_0 由 make_q8.sh 生成)

[ -f /tmp/hexenv.sh ] && source /tmp/hexenv.sh 2>/dev/null   # 本机 adb / Hexagon SDK 路径(如有)

export DEV=/data/local/tmp/llamahtp   # 设备上的部署目录
# DSP 库搜索路径:. = 本目录(找 htp skel),外加系统 adsp 路径
export ADSP="ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp"

# 在设备上、以正确的库路径、从部署目录运行一条命令
hexrun() {
  adb shell "cd $DEV && LD_LIBRARY_PATH=. $ADSP $*" 2>/dev/null | tr -d '\r'
}

# 杀掉残留的 llama 进程
hexkill() { adb shell "pkill -9 -f llama" 2>/dev/null; sleep 1; }
