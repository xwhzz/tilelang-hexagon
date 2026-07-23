# Copy to config.sh and adjust paths for the demo machine.
export TILELANG_ROOT=/path/to/tilelang-hexagon
export TILELANG_PYTHON=/path/to/conda/env/bin/python
export LLAMA_CPP_ROOT=/path/to/llama.cpp
export LLAMA_HTP_BUILD=/path/to/llama.cpp/build-snap/ggml/src/ggml-hexagon/htp-v79-prefix/src/htp-v79-build
export CMAKE_BIN=cmake
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.6.0.0
export HEXAGON_TOOLS_ROOT=/path/to/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07
export ANDROID_NDK_ROOT=/path/to/android-ndk-r25c

export DEVICE_SERIAL=
export DEVICE_DIR=/data/local/tmp/llamahtp
export MODEL_Q4=LFM2-1.2B-Q4_0.gguf
export MODEL_Q8=LFM2-1.2B-Q8_0.gguf
export DSP_ARCH=v79
