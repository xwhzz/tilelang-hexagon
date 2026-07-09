#pragma once
// Common header for tilelang-generated Hexagon cDSP kernels.
//
// Generated kernels are compiled as C++ by hexagon-clang++ from the Hexagon
// SDK.  This header is kept SDK-light: it only provides the scalar/vector type
// vocabulary the codegen emits.  HMX/HVX/VTCM helpers (which pull in SDK and
// intrinsic headers) live in sibling headers included on demand by kernels that
// use those features.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

// Hexagon has a native IEEE half type; back the codegen's `half` with it so no
// software emulation is needed on device.
using half = __fp16;

// Kernel status ABI.  A tilelang Hexagon kernel entry returns int32_t — 0 on
// success, nonzero on an unrecoverable device condition.  The FastRPC skel maps
// nonzero to AEE_EFAILED (tl_hexagon_forward_kernel_status in _fastrpc.py), so
// the host's run() raises instead of silently returning unwritten/partial
// output.  The codegen emits these codes into the kernel/worker source.
#define TL_OK 0
#define TL_ERR_VTCM 1 // VTCM region unavailable / no worker region fits the grant
#define TL_ERR_HMX 2  // HMX could not be enabled for a worker thread

#ifndef TL_DEVICE
#define TL_DEVICE static inline __attribute__((always_inline))
#endif

// Fixed-width vector vocabulary backing the `floatN`/`halfN`/`intN` spellings
// the C codegen emits (e.g. `*(float4*)(ptr + off)`, `(half4)(x)`, `a * b`).
//
// Backed by a NATIVE clang ext_vector so element-wise ARITHMETIC lowers to real
// HVX ops (vand/vlsr/vmpy/…) instead of scalar loops — a DSL dequant like
// `(int16)(q & 0xF) - 8) * s` becomes native vector `&`/`-`/`*` that
// hexagon-clang++ maps onto HVX.  `aligned(1)` keeps the codegen's reinterpret
// loads/stores (`*(halfN*)(ptr + off)`) unaligned-safe — the VTCM shared-memory
// merge pass packs tiles at element, not 128-byte, offsets; the value ops
// themselves are alignment-independent, so this costs only the load/store form
// (vmemu vs vmem), never vectorization.
template <typename T, int N> struct vec_type {
  typedef T nat_t __attribute__((ext_vector_type(N)));
  nat_t v __attribute__((aligned(1)));
  vec_type() = default;
  vec_type(nat_t n) : v(n) {} // wrap a native result (operator returns / loads)
  // Broadcast ctor.  const-ref because __fp16 is not a valid by-value parameter
  // type on Hexagon.  `(nat_t)scalar` is clang's ext_vector splat.
  explicit vec_type(const T &s) { v = (nat_t)s; }
  // Converting ctor for vector casts, e.g. (half4)float4_value when storing an
  // fp32 accumulator back as fp16 — a lane-wise numeric convert.
  template <typename U> explicit vec_type(const vec_type<U, N> &o) {
    v = __builtin_convertvector(o.v, nat_t);
  }
  // Element-wise operators -> native (HVX) vector ops.  Members are instantiated
  // lazily, so bit-ops on float aggregates (never emitted) never get type-checked.
#define TL_VEC_BINOP(op)                                                        \
  vec_type operator op(const vec_type &o) const { return vec_type(v op o.v); }
  TL_VEC_BINOP(+) TL_VEC_BINOP(-) TL_VEC_BINOP(*) TL_VEC_BINOP(/)
  TL_VEC_BINOP(&) TL_VEC_BINOP(|) TL_VEC_BINOP(^) TL_VEC_BINOP(<<) TL_VEC_BINOP(>>)
#undef TL_VEC_BINOP
};

// Widths up to 128 lanes: HVX is a 1024-bit vector, so one register holds 128
// int8 / 64 fp16 / 32 fp32.  The codegen's Hexagon vectorizer plans up to that.
#define TL_DEFINE_VEC(T)                                                        \
  using T##2 = vec_type<T, 2>;                                                  \
  using T##4 = vec_type<T, 4>;                                                  \
  using T##8 = vec_type<T, 8>;                                                  \
  using T##16 = vec_type<T, 16>;                                                \
  using T##32 = vec_type<T, 32>;                                                \
  using T##64 = vec_type<T, 64>;                                                \
  using T##128 = vec_type<T, 128>;

TL_DEFINE_VEC(float)
TL_DEFINE_VEC(half)
TL_DEFINE_VEC(double)
TL_DEFINE_VEC(int8_t)
TL_DEFINE_VEC(int16_t)
TL_DEFINE_VEC(int32_t)
TL_DEFINE_VEC(int64_t)
TL_DEFINE_VEC(uint8_t)
TL_DEFINE_VEC(uint16_t)
TL_DEFINE_VEC(uint32_t)
TL_DEFINE_VEC(uint64_t)
