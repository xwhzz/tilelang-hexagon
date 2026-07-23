# Hexagon 开发环境配置(从零搭建)

这台开发主机(`/home/xwh`)是**准裸机**:只有 `/usr/bin/python3`(无 pip/conda),无系统级
`cmake`/`ninja`/`make`/`gcc`/`clang`,**无免密 sudo**(apt 不能非交互用),内存约 **5.4 GB
(可用 ~3.3 GB)**。本文记录已经搭好的、可复现的完整环境,覆盖三个构建目标:

1. **tilelang 原生库**(x86,改 C++ codegen 后要重编)—— 用 conda 环境
2. **mini-htp / DSP skel**(Hexagon,`hexagon-clang`)—— 用 Hexagon SDK
3. **llama.cpp 的 ggml-hexagon skel**(Hexagon)—— 用 SDK + NDK,是本轮性能对接的主战场

> ⚠️ **已知内存瓶颈(务必先读)**:本机 ~3.3 GB 可用内存,**编不动某些大生成 kernel**。
> `kernel_qmatmul_hmx_atoms_32x128x8192.cc`(K=8192,把 256 个 K-tile 的 Crouton 全 materialize
> 在一个函数里)在 `hexagon-clang -O2` 下需 **>4.5 GB** —— 无上限时会吃满 swap 把整机冻死。
> 详见文末「内存限制」。**要编这类 kernel,请用 ≥16 GB 内存的机器,或先把 kernel 改成流式 K。**

---

## 0. 关键路径速查

| 组件 | 版本 | 路径 |
|---|---|---|
| Hexagon SDK | 6.6.0.0 | `/home/xwh/Downloads/Hexagon_SDK_Linux/Hexagon_SDK/6.6.0.0` |
| hexagon-clang(DSP 工具链) | 19.0.07 | `$SDK/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang++` |
| Android NDK | r25c | `/home/xwh/Downloads/android-ndk-r25c-linux/android-ndk-r25c` |
| 便携 cmake + ninja | 3.30.5 | `/home/xwh/.local/cmake/`(`bin/cmake`, `bin/ninja`) |
| conda(Miniforge)| — | `/home/xwh/miniforge3`,环境 `tl`(python 3.11.15) |
| tilelang 仓库 | 分支 `hexagon-backend` | `/home/xwh/tilelang-hexagon` |
| llama.cpp 仓库 | commit `4fc4ec55` | `/home/xwh/scratchpad-llamacpp`(构建目录 `build-snap/`) |
| 环境一键脚本 | — | `/tmp/hexenv.sh`(session 级临时文件,见 §3) |
| 设备 | OnePlus 13 · PJZ110 · Hexagon **v79**(SM8750)| adb 已授权;部署目录 `/data/local/tmp/llamahtp` |

---

## 1. 主机基础工具(无 sudo 的绕法)

系统没有 cmake/ninja/make/编译器,全部装在用户目录:

```bash
# 便携 cmake(Kitware GitHub release;cmake.org 会 403,用 GitHub)
mkdir -p ~/.local && cd ~/.local
curl -L https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-x86_64.tar.gz | tar xz
mv cmake-3.30.5-linux-x86_64 cmake
# 把 ninja 放进 cmake/bin(SDK 的 build_cmake 宏期望 CMAKE_ROOT_PATH 下有 ninja)
cp <你的 ninja> ~/.local/cmake/bin/ninja
# 有些 SDK 脚本调裸 `python`
mkdir -p ~/.local/bin && ln -sf /usr/bin/python3 ~/.local/bin/python
export PATH=~/.local/cmake/bin:~/.local/bin:$PATH
```

---

## 2. Hexagon SDK

解压到 `Downloads/Hexagon_SDK_Linux/`。SDK 自带 `hexagon-clang 19.0.07` 和 DSP runtime。

**一个必须修的坑 —— qaic 符号链接**:cmake 的 `build_idl` 宏要 `qaic` 在
`ipc/fastrpc/qaic/bin/qaic`,而预编译的在 `ipc/fastrpc/qaic/Ubuntu/qaic`:

```bash
SDK=/home/xwh/Downloads/Hexagon_SDK_Linux/Hexagon_SDK/6.6.0.0
ln -sf ../Ubuntu/qaic $SDK/ipc/fastrpc/qaic/bin/qaic   # 或直接 cp 过去
```

SDK 的 env 脚本:`$SDK/setup_sdk_env.source`。它会 `unset` 后重设
`HEXAGON_SDK_ROOT`/`HEXAGON_TOOLS_ROOT` 等。**注意**:如果 `HEXAGON_SDK_ROOT` 已经预设,
脚本会 bail —— 所以要先 `unset HEXAGON_SDK_ROOT` 再 source(§3 的脚本已处理)。

source 时会打印两条 **非致命** 警告,忽略即可:
- `Failed to install QAIC`(本机无 `make`,但用的是预编译 qaic)
- `python not available`(用 Ninja generator,不需要)

---

## 3. `/tmp/hexenv.sh` —— 环境一键脚本

这是最常用的入口:`source /tmp/hexenv.sh` 就得到完整构建环境。它是 **session 级临时文件**
(重开会话会没),内容如下,按需重建:

```bash
cat > /tmp/hexenv.sh <<'EOF'
# Hexagon 构建环境
unset HEXAGON_SDK_ROOT                      # setup 脚本预设了会 bail
source /home/xwh/Downloads/Hexagon_SDK_Linux/Hexagon_SDK/6.6.0.0/setup_sdk_env.source
export CMAKE_ROOT_PATH=/home/xwh/.local/cmake          # build_cmake 宏期望的布局
export PATH=/home/xwh/.local/cmake/bin:/home/xwh/.local/bin:$PATH
export ANDROID_ROOT_DIR=/home/xwh/Downloads/android-ndk-r25c-linux/android-ndk-r25c
export ANDROID_NDK_ROOT=/home/xwh/Downloads/android-ndk-r25c-linux/android-ndk-r25c
# adb(platform-tools)也要在 PATH 上
EOF
```

---

## 4. Android NDK r25c

解压到 `Downloads/android-ndk-r25c-linux/`。用于编 HLOS(Android arm64)侧:llama.cpp 的
host 库/可执行、mini-htp 的 host 测试程序。`ANDROID_NDK_ROOT` 由 `/tmp/hexenv.sh` 设好。

---

## 5. conda 环境 `tl` —— 编 tilelang 原生库

基础主机没有 x86 C++ 工具链/pip,所以 tilelang 的原生库(TVM + tilelang,scikit-build)
在 Miniforge 环境里从源码编。

```bash
# 一次性
conda create -n tl -c conda-forge python=3.11 c-compiler cxx-compiler cmake ninja zlib make
#   ↑ make 必须有:TVM 的 libbacktrace ExternalProject 用 configure+make,否则 ~第 11 个对象就死
conda activate tl
pip install --index-url https://download.pytorch.org/whl/cpu torch      # CPU torch
pip install numpy cython scikit-build-core "z3-solver>=4.13,<4.15.5" patchelf \
            apache-tvm-ffi==0.1.11 cloudpickle ml-dtypes psutil tqdm typing-extensions
```

> ⚠️ **必须 pin `apache-tvm-ffi==0.1.11`**(vendored `3rdparty/tvm/3rdparty/tvm-ffi` 是 0.1.11)。
> pip 默认拉 0.1.12,其 `libtvm_ffi.so` 会和 tilelang 自己编的冲突 → `import tilelang` 崩:
> `TypeAttr __ffi_repr__ is already registered for type index 130`。

**首次全量编**:
```bash
conda activate tl && cd /home/xwh/tilelang-hexagon
CMAKE_POLICY_VERSION_MINIMUM=3.5 CMAKE_BUILD_PARALLEL_LEVEL=2 \
  pip install -e . --no-build-isolation
```
- CUDA/ROCm 自动 OFF(FindPipCUDAToolkit 找不到 → CPU-only)。
- **`-j2` 别加大**:只有 ~3.2 GB 空闲,`-j4` 会 OOM。

**改了 C++(如 `src/hexagon/...`)后增量重编**:
```bash
conda activate tl
CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake --build build -j2   # 就地重链 build/lib/libtilelang.so
```
Ninja 会在 CMakeLists 变化时自动 reconfigure。**跑任何 tilelang 都要先 `conda activate tl`。**

---

## 6. 设备(adb + FastRPC)

- 设备:OnePlus 13(PJZ110),Hexagon **v79**。adb 已授权。
- 部署目录:`/data/local/tmp/llamahtp`(放 skel/二进制/模型/lib)。
- DSP 库搜索路径(跑时设):`ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp`。
- **USB 偶尔掉线**(本轮多次)。重连:
  ```bash
  adb kill-server; adb start-server; adb wait-for-device
  ```

---

## 7. 三个构建目标

### 7a. mini-htp(参考实现,验证 SDK 通路)
```bash
source /tmp/hexenv.sh
cd /home/xwh/tilelang-hexagon/mini-htp
build_cmake hexagon DSP_ARCH=v79   # -> hexagon_*_v79/libhmx_matmul_skel.so(DSP,-mhmx)
build_cmake android                # -> android_*_aarch64/hmx_matmul_test(host,NDK)
# 推 skel + host exe 到 /data/local/tmp/hmx_matmul,带 ADSP_LIBRARY_PATH 跑
```

### 7b. llama.cpp 的 ggml-hexagon skel(本轮主战场)
```bash
source /tmp/hexenv.sh
LCPP=/home/xwh/scratchpad-llamacpp
cp $LCPP/../<presets>/CMakeUserPresets.json $LCPP/   # 已在:arm64-android-snapdragon-release
cmake --preset arm64-android-snapdragon-release -B $LCPP/build-snap -DGGML_OPENCL=OFF
cmake --build $LCPP/build-snap --target llama-cli htp-v79 -j$(nproc)   # 基线
adb push $LCPP/build-snap/ggml/src/ggml-hexagon/libggml-htp-v79.so /data/local/tmp/llamahtp/
```
改了 host 侧 `ggml-hexagon.cpp`(如 lm_head 上限):`--target ggml-hexagon`,推
`libggml-hexagon.so`。改了 skel 侧:`--target htp-v79`,推 `libggml-htp-v79.so`。

**tilelang 算子对接进 skel**(详见 `examples/hexagon/llama_cpp_integration/`):把生成的
`kernel_*.c/.cc` + `tl_ggml_*.cc` 拷进 `$LCPP/ggml/src/ggml-hexagon/htp/`,`git apply` 对应
patch(`ggml-hexagon.patch` = Q4 HMX prefill;`ggml-hexagon-q8.patch` = Q8 decode GEMV),
CMake 会把 tilelang 模板路径(`/home/xwh/tilelang-hexagon/src`)加进 include。**两个集成的
registry 不能同时编进一个 skel。**

### 7c. tilelang 原生库
见 §5。

---

## 8. 内存限制(本轮踩的大坑,务必知道)

本机 ~3.3 GB 可用 + 4 GB swap(~2.5 GB 空)。以下会 **OOM / 冻机**:

- **`hexagon-clang -O2` 编大 HMX-atom kernel**:`kernel_qmatmul_hmx_atoms_32x128x8192.cc`
  (K=8192)需 **>4.5 GB**,`...x2048.cc`(K=2048)也 >4.5 GB —— 因为 kernel 把 128/256 个
  K-tile 的 Crouton **全 materialize 在一个函数里**,codegen 的指令调度+寄存器分配对这么多
  同时 live 的值内存超线性膨胀。无上限时吃满 swap 把整机拖死。
  - **这跟 C-codegen 选型无关** —— 换 TVM 的 LLVM 路径也是同一个 Hexagon 后端做调度/分配,同样炸。
  - **真正的修法**:把 kernel 改成流式 K(只保留 1–2 个 live Crouton),这是设计文档里的 future work。
  - **临时绕法**:`-O0` 能秒编过(105 MB),但 dequant 退化成标量 → prefill 大幅倒退,性能没意义。
  - **正解**:在 ≥16 GB 内存的机器上编(kernel 本身在大内存机上 `-O2` 没问题)。

- **调试单个 kernel 的编译**(不冻整机):从 `build-snap/.../htp-v79-build/compile_commands.json`
  取确切命令,`ulimit -v 3145728` 加内存上限单独跑,`/usr/bin/time -v` 看峰值 RSS。

- **其它并行度**:tilelang 原生库 `-j2`;skel 里那两个大 kernel 用 `-j1`(每个 codegen ~2 GB)。

---

## 9. 常见排障速查

| 症状 | 原因 / 解 |
|---|---|
| `setup_sdk_env.source` bail | `HEXAGON_SDK_ROOT` 预设了 → 先 `unset`(hexenv.sh 已做) |
| `build_idl` 找不到 qaic | 建 `ipc/fastrpc/qaic/bin/qaic` 符号链接(§2) |
| `import tilelang` 崩 `__ffi_repr__ ... registered` | `pip install apache-tvm-ffi==0.1.11`(§5) |
| tilelang 编译 OOM | 用 `-j2`,别 `-j4`(§5) |
| skel 大 kernel 编译冻机 | 内存不够,见 §8;换大内存机或改流式 kernel |
| `adb devices` 空 | USB 掉线 → `adb kill-server; adb start-server; adb wait-for-device`(§6) |
| TVM libbacktrace 编译中途死 | conda 环境缺 `make`(§5) |
| DSP 库找不到 | 跑时 `ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp` |
