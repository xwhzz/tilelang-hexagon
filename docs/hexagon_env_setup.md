# Hexagon 开发环境配置(从零搭建)

这台开发主机(`/home/xwh`)是**准裸机**:只有 `/usr/bin/python3`(无 pip/conda),无系统级
`cmake`/`ninja`/`make`/`gcc`/`clang`,**无免密 sudo**(apt 不能非交互用),内存约 **5.4 GB
(可用 ~3.3 GB)**。本文记录已经搭好的、可复现的完整环境,覆盖两个构建目标:

1. **tilelang 原生库**(x86,改 C++ codegen 后要重编)—— 用 conda 环境
2. **DSP skel 与 Android host 程序**(Hexagon,`hexagon-clang`;arm64,NDK)—— 用 Hexagon SDK,
   由 tilelang 的运行时在 `tilelang.compile` 时自动构建

> ⚠️ **已知内存瓶颈(务必先读)**:本机 ~3.3 GB 可用内存,**编不动特别大的生成 kernel**
> (`hexagon-clang -O2` 对上千行、大量同时 live 值的函数会用到数 GB)。详见文末「内存限制」。

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
| 环境一键脚本 | — | `/tmp/hexenv.sh`(session 级临时文件,见 §3) |
| 设备 | OnePlus 13 · PJZ110 · Hexagon **v79**(SM8750)| adb 已授权;运行时部署到 `/data/local/tmp/tilelang_hexagon` |

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

解压到 `Downloads/android-ndk-r25c-linux/`。用于编 HLOS(Android arm64)侧的 host 程序
(运行时的 FastRPC agent)。`ANDROID_NDK_ROOT` 由 `/tmp/hexenv.sh` 设好。

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
- 部署目录:运行时自动推送 skel 和 agent 到 `/data/local/tmp/tilelang_hexagon`。
- DSP 库搜索路径(跑时设):`ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp`。
- **USB 偶尔掉线**(本轮多次)。重连:
  ```bash
  adb kill-server; adb start-server; adb wait-for-device
  ```

---

## 7. 内存限制(务必知道)

本机 ~3.3 GB 可用 + 4 GB swap(~2.5 GB 空)。以下会 **OOM / 冻机**:

- **`hexagon-clang -O2` 编特别大的 kernel**:把很多 tile 全部展开在一个函数里、大量值同时
  live 的 kernel,指令调度和寄存器分配的内存会超线性膨胀,超过 4 GB 时吃满 swap 把整机拖死。
  这跟 C codegen 无关,LLVM 路径用的是同一个 Hexagon 后端。修法是让 kernel 流式处理(只保留
  少量 live 值),或在 ≥16 GB 内存的机器上编。
- **调试单个 kernel 的编译**(不冻整机):`ulimit -v 3145728` 加内存上限单独跑生成的
  `_dsp.cc` 编译命令,`/usr/bin/time -v` 看峰值 RSS。
- **其它并行度**:tilelang 原生库 `-j2`。

---

## 8. 常见排障速查

| 症状 | 原因 / 解 |
|---|---|
| `setup_sdk_env.source` bail | `HEXAGON_SDK_ROOT` 预设了 → 先 `unset`(hexenv.sh 已做) |
| `build_idl` 找不到 qaic | 建 `ipc/fastrpc/qaic/bin/qaic` 符号链接(§2) |
| `import tilelang` 崩 `__ffi_repr__ ... registered` | `pip install apache-tvm-ffi==0.1.11`(§5) |
| tilelang 编译 OOM | 用 `-j2`,别 `-j4`(§5) |
| 大 kernel 编译冻机 | 内存不够,见 §7;换大内存机或改流式 kernel |
| `adb devices` 空 | USB 掉线 → `adb kill-server; adb start-server; adb wait-for-device`(§6) |
| TVM libbacktrace 编译中途死 | conda 环境缺 `make`(§5) |
| DSP 库找不到 | 跑时 `ADSP_LIBRARY_PATH=.:/vendor/lib/rfsa/adsp:/dsp` |
