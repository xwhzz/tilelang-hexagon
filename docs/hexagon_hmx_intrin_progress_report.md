# TileLang Hexagon HMX Intrinsic 与原生 Crouton GEMM 进展报告

## 1. 基本信息

- 日期：2026-08-20
- 开发分支：`feat/hexagon-hmx-intrin-layout`
- commit：`3268f8292f5df0ce519da260a026fd8ad69b7c2a`
- commit message：`feat(hexagon): add native HMX Crouton copy and GEMM lowering`
- 基线 commit：`c303ff8e11fa45a79bc4f70c2742010a15673a4c`
- 改动规模：25 个文件，新增 1423 行，删除 361 行

## 2. 本阶段目标与结论

本阶段主要完成两项工作：

1. 实现面向 Qualcomm HMX 的 instruction-level `HMXIntrinEmitter`，用 TileLang macro
   显式表达 HMX 的 accumulator 协议和 32x32x32 MAC atom。
2. 在该 emitter 基础上实现 Hexagon `T.gemm` lowering，使 `T.gemm` 的公开接口只接收
   VTCM 中的 A/B/C buffer，并由 layout inference 和 lowering 负责原生 Crouton layout、
   HMX 指令循环以及内部状态。

同时补充了 row-major DDR/VTCM 与原生 Crouton VTCM 之间的专用 `T.copy` lowering。
满足条件时，用户只写一次逻辑 `T.copy`，后端直接生成带基址和 stride 的
pack/unpack helper，不需要在 DSL 中显式写“DDR -> row-major VTCM -> Crouton VTCM”两级
buffer。

当前已经打通并在 Hexagon v79 真机验证的主链路为：

```text
row-major DDR matrix/slice
  -> T.copy
  -> tl_hexagon_hmx_pack_crouton
  -> native Crouton A/B in VTCM
  -> T.gemm
  -> acquire -> clear -> mma_atom* -> convert -> store -> release
  -> native Crouton C in VTCM
  -> T.copy
  -> tl_hexagon_hmx_unpack_crouton
  -> row-major DDR output
```

当前 `T.copy` 是同步 HVX/scalar copy 和 layout transform，不包含 DMA。接口中保留了
row-major 端的基址和 stride，因此后续可在不改变 DSL 和 `T.gemm` 接口的情况下引入
DMA staging。

## 3. 核心设计

### 3.1 `T.Layout` 与 Crouton 数据变换的边界

`T.Layout` 负责把逻辑矩阵坐标映射为 Crouton 物理地址，不负责实际搬运、pack/unpack、
VTCM 分配、对齐或 HMX 状态管理。

FP16 HMX 的 32x32 Crouton atom 共 1024 个 FP16 元素，即 2048 bytes。atom 内地址为：

```text
cpos(row, col) = (row // 2) * 64 + col * 2 + row % 2
```

在此 atom 上使用 `Layout.repeat` 和 `Layout.expand` 组合完整矩阵布局：

| 操作数 | 逻辑矩阵 | Crouton tile 顺序 | 主要对齐要求 |
|---|---|---|---|
| Activation A | `[M, K]` | `[M_tile, K_tile, cpos]` | 2048 B |
| Weight B | `[K, N]` | `[N_tile, K_tile, cpos]` | 128 B |
| Output C | `[M, N]` | `[M_tile, N_tile, cpos]` | 2048 B |
| Scale/bias config | 64 个 `uint32` | scale 32 words + bias 32 words | 256 B |

A/C 和 B 的 tile 顺序不同，不能用同一个通用“swizzled matrix”布局代替。A/B 的逻辑
transpose 通过 layout composition 表达，HMX emitter 仍按数学意义上的 A `[M,K]` 和
B `[K,N]` 选择 tile。

### 3.2 `HMXIntrinEmitter`

主要实现位于 `tilelang/hexagon/hmx_intrin.py`。提供以下原子接口：

- `activation_layout()`、`weight_layout()`、`output_layout()`：返回操作数对应的原生
  Crouton layout。
- `acquire()` / `release()`：取得和释放唯一 HMX accumulator 的使用权。
- `clear()`：清空隐式 accumulator。
- `load_bias()`：加载 convert 使用的 scale/bias config。
- `mma_atom()`：发射一个 32x32x32 HMX MAC atom。
- `mma_tile()`：遍历全部 K atom，计算一个 `(M_tile, N_tile)` 输出 tile。
- `convert()`：把隐式 accumulator 转换为 FP16 convert state。
- `store()`：直接写入指定的原生 Crouton C tile。

HMX 和 CUDA tensor core 的关键差异是：HMX 没有可寻址的 accumulator fragment 数组，
而是每个 HMX context 只有一个隐式 accumulator；A/B 的 paired `mxmem` load 本身会触发
MAC。因此 emitter 不能像 CUDA emitter 那样同时维护多个输出 fragment，必须按下面的
顺序串行完成每个输出 tile：

```python
emitter.acquire(acc)
for mt in T.serial(M // 32):
    for nt in T.serial(N // 32):
        emitter.clear(acc)
        emitter.load_bias(bias, bias_vtcm)
        for kt in T.serial(K // 32):
            emitter.mma_atom(acc, A_hmx, B_hmx, mt, nt, kt)
        emitter.convert(cvt, acc, bias, bias_vtcm)
        emitter.store(cvt, acc, C_hmx, bias, bias_vtcm, mt, nt)
emitter.release(acc)
```

`acc_state`、`cvt_state` 和 `bias_state` 是用于 TIR dependency/liveness 的 token，不是
实际可读写的寄存器 fragment。A/B 的生命周期在最后一个读取它们的 HMX multiply packet
完成后结束；同一 Q6 thread 上后续 C++ 语句可以安全覆盖其 VTCM。`store()` 因此只保留
convert state、accumulator state、bias/config 和 C 的依赖，不再人为把 A/B 延长到输出
store。

最终实现允许 HMX convert state 直接 store 到目标 C tile。因为每个输出 tile 本身就是
2 KiB 且 C 基址按 2 KiB 对齐，所以不再分配固定 `output_atom`，也不需要额外的 HVX
commit/copy。

### 3.3 `T.gemm` 的 HMX lowering

主要实现位于 `tilelang/hexagon/gemm_hmx.py` 和 `src/hexagon/op/gemm.cc`。

`GemmHMX.infer_layout()` 为 A/B/C shared buffer 分别推断 activation、weight 和 output
Crouton layout。`GemmHMX.lower()` 内部分配 scale/bias config 和三个 dependency token，
并生成 M/N/K tile 循环。因此 DSL 侧只需要：

```python
A_hmx = T.alloc_shared((BM, K), "float16")
B_hmx = T.alloc_shared((K, BN), "float16")
C_hmx = T.alloc_shared((BM, BN), "float16")
T.copy(A[by * BM, 0], A_hmx)
T.copy(B[0, bx * BN], B_hmx)
T.gemm(A_hmx, B_hmx, C_hmx, clear_accum=True)
T.copy(C_hmx, C[by * BM, bx * BN])
```

新路径不再调用 monolithic `tl_hexagon_hmx_gemm`。当前 selector 只在以下条件全部满足时
选择 `hexagon.hmx`：

- A/B/C 均为 FP16；
- M/N/K 均为 32 的倍数；
- A/B/C 均为完整的二维 shared/VTCM buffer；
- `clear_accum=True`；
- GEMM region 覆盖各自完整 buffer。

不满足条件的情况继续走原来的 `cpu.scalar` fallback，避免扩大本阶段实现的语义范围。
当前 HMX 路径支持 A/B logical transpose，但只支持 FP16 输出和 overwrite GEMM。

### 3.4 一次逻辑 `T.copy` 完成 row-major/Crouton 转换

主要实现位于 `src/hexagon/op/copy.cc` 和 `src/tl_templates/hexagon/hmx.h`。

copy lowering 会检查 layout 的实际 flatten offset，而不是只比较 layout 输出 rank。这是
因为 `Layout.repeat(..., factor=1)` 会消去 singleton tile axis；例如 M=32 的 qmatmul
仍应被识别为合法 activation layout。

专用 lowering 的匹配条件为：

- src/dst 均为 FP16；
- 恰有一端是带已识别 HMX layout 的 shared/VTCM buffer；
- HMX 端是完整、静态、二维且两个维度都是 32 的倍数；
- row-major 端可以是 global/DDR 或 shared/VTCM，也可以是更高维 tensor 中的矩阵 slice；
- row-major 端前导维度的 region extent 为 1，最后两维与 HMX buffer shape 一致。

lowering 根据矩阵 slice 计算真实 base pointer 和最后两维 stride，然后生成一次：

```text
tl_hexagon_hmx_pack_crouton(dst, src, rows, cols, stride0, stride1, layout_kind)
```

或：

```text
tl_hexagon_hmx_unpack_crouton(dst, src, rows, cols, stride0, stride1, layout_kind)
```

layout kind 包括 activation、weight 以及二者的 transposed 形式。常见的非转置、64 列
对齐形式使用 HVX `vshuff`/`vdeal`；转置形式和不满足 HVX 快路径的宽度使用 scalar
correctness fallback。无法匹配的 copy 继续走 `LowerNormalCopy`。

这里的“融合”指一个 TileLang copy op 同时完成数据搬运和 layout transform，不表示已经
使用硬件 DMA，也不表示所有形状都能走 HVX 快路径。

### 3.5 VTCM 对齐传播

`src/transform/merge_shared_memory_allocations.cc` 增加了按 HMX operand role 传播对齐的逻辑：

- `hexagon_hmx_mma` 的 activation：2048 B；
- `hexagon_hmx_mma` 的 weight：128 B；
- `hexagon_hmx_store` 的 output：2048 B；
- load-bias/convert/store 使用的 config：256 B。

同一个 shared allocation 被多个路径使用时取最大对齐要求。该实现没有把所有 HMX
buffer 一律提升到 2 KiB，避免 weight/config 产生不必要的 VTCM padding。

## 4. 配套改动

### 4.1 示例迁移

- `example_matmul.py`：A/B 由 DDR 一次 `T.copy` 到原生 Crouton VTCM，C 一次 copy 回 DDR。
- `example_qmatmul.py`：A/C 使用直接 Crouton copy；B 的 HVX dequant producer 仍先写
  row-major VTCM，再用一次 `T.copy` pack 到 weight Crouton。
- `example_worker_pool.py`：每个 worker 使用自己的 Crouton VTCM slice；HMX accumulator
  仍由单一 HMX lock 串行保护。
- `example_qmatmul_kstream.py`：适配统一的 `mma_tile()`、prefix 和标准二维 Crouton 坐标。
- `example_flash_attention.py`：显式保留 HMX Crouton buffer 与 HVX row-major reduce/map
  buffer 之间的 pack/unpack 边界。
- `offline_matmul`：生成代码和 FastRPC DSP wrapper 更新为新的 explicit-atom 路径。

### 4.2 Codegen、runtime 与文档

- Hexagon codegen 增加 explicit HMX intrinsic 的生成和 worker-pool allowlist 支持。
- `hmx.h` 增加 strided pack/unpack、非对齐 HVX load/store 以及 scalar fallback。
- `_fastrpc.py` 修正 HMX/Q4 wrapper 所需 template header 的选择。
- backend 中英文文档、DSL 文档、compiler walkthrough 和 offline README 已同步更新。

## 5. 验证结果

### 5.1 Host/codegen 测试

提交前执行：

```bash
source /home/lyn/workspace/hexagon-env.sh
TILELANG_CACHE_DIR=/tmp/tilelang-cache-hmx-commit \
pytest -q \
  testing/python/hexagon/test_tilelang_hmx_intrin.py \
  testing/python/hexagon/test_tilelang_qgemv.py::test_q8_0_tiled_layout_contract
```

结果：

```text
18 passed, 2 warnings
```

覆盖内容包括：

- Crouton atom、矩形矩阵、transpose 和 singleton tile axis 的 layout contract；
- 标准 `(mt, nt, kt)` 坐标到 HMX tile 地址的映射；
- `T.gemm` 选择、layout inference、explicit atom lowering 和 scalar fallback；
- A/B/C/config 按角色对齐；
- DDR matrix slice 的单次 pack/unpack helper；
- qmatmul dequant staging、worker-local copy 和 flash-attention row-reduce 边界；
- Q4 K-streaming、FastRPC wrapper、embeddable manifest 和 checked-in dispatch contract。

此外以下构建已通过：

```text
cmake --build build -j2
build_cmake hexagon DSP_ARCH=v79
build_cmake android
git diff --check
```

### 5.2 Hexagon v79 真机验证

测试设备为 OnePlus 13 / SM8750 / Hexagon v79：

| 用例 | 形状/场景 | 结果 |
|---|---|---|
| Offline FP16 matmul | 256x256x256，DDR/Crouton fused copy + `T.gemm` | max abs err 约 `9.77e-4`，PASS |
| Q4 matmul | 32x128x128，A/C direct copy，B dequant staging | relative error `0.000351`，PASS |
| Worker-pool batched matmul | worker-local Crouton buffers | max abs err `0.0009747`，PASS |
| Transposed-B probe | 64x64x64 | max abs err `0.0006509`，PASS |

这些结果证明当前链路已经完成数值闭环，但不构成性能结论。当前 copy 仍是同步
HVX/scalar 实现，单次 FastRPC wall time 也包含 marshaling 和启动开销。

## 6. 当前限制与风险

1. **尚未支持 DMA。** 当前一次 `T.copy` 由同步 helper 完成 DDR/VTCM 搬运和 Crouton
   变换；后续若接入 DMA，较现实的路径仍可能是 DMA 到 row-major VTCM staging，再由
   HVX pack 到 Crouton，或者为满足对齐/stride 的形状增加专用融合路径。
2. **`T.gemm` 支持面有限。** 目前只支持完整二维 shared FP16 buffer、32 倍数尺寸、
   FP16 输出和 `clear_accum=True`。partial region、动态尺寸、混合 dtype 和 accumulator
   preload/追加累加仍走 fallback。
3. **转置 pack 尚未 HVX 优化。** `transpose_A/B` 的功能路径正确，但当前使用 scalar
   pack，是明确的性能待办。
4. **FlashAttention 仍需要 layout boundary。** HVX rowmax/rowsum 需要 row-major `S`，
   因此当前保留 `S_hmx <-> S` 和 `temp_hmx -> temp`。曾尝试删除这些 staging，但真机
   数值误差达到 `0.121`，因此该实验没有进入 commit。
5. **尚未完成流水与性能优化。** DMA queue、double buffering、copy/HVX 与 HMX overlap、
   transposed fast path 和 autotune 均不在本 commit 范围内。
6. **旧 monolithic helper 尚未删除。** 新 `T.gemm` 已不再依赖
   `tl_hexagon_hmx_gemm`，但旧 helper 仍用于兼容既有入口；后续可在调用点完全迁移后再
   单独清理。

## 7. 主要代码入口

| 文件 | 作用 |
|---|---|
| `tilelang/hexagon/hmx_intrin.py` | Crouton layout 与 `HMXIntrinEmitter` |
| `tilelang/hexagon/gemm_hmx.py` | `T.gemm` layout inference 和 explicit HMX lowering |
| `src/hexagon/op/gemm.cc` | Hexagon GEMM instruction selection 与 fallback |
| `src/hexagon/op/copy.cc` | row-major/Crouton `T.copy` 识别和 lowering |
| `src/tl_templates/hexagon/hmx.h` | HMX atom wrapper、strided pack/unpack 和 HVX/scalar 实现 |
| `src/transform/merge_shared_memory_allocations.cc` | HMX operand role 对齐传播 |
| `testing/python/hexagon/test_tilelang_hmx_intrin.py` | layout、lowering、copy、对齐和集成 contract 测试 |
