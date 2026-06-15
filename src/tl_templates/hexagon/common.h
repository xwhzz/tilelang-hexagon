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

#ifndef TL_DEVICE
#define TL_DEVICE static inline __attribute__((always_inline))
#endif

// Fixed-width vector vocabulary backing the `floatN`/`halfN`/`intN` spellings
// the C codegen emits (e.g. `*(float4*)(ptr + off)`).  These are plain
// aggregates; hexagon-clang++ lowers element-wise loops over them and
// auto-vectorizes onto HVX where profitable.
template <typename T, int N> struct vec_type {
  T data[N];
  vec_type() = default;
  // Broadcast ctor.  Value taken by const-ref because __fp16 is not a valid
  // by-value function parameter type on Hexagon.
  explicit vec_type(const T &v) {
    for (int i = 0; i < N; ++i) data[i] = v;
  }
  // Converting ctor for vector casts, e.g. (half4)float4_value when storing an
  // fp32 accumulator back as fp16.
  template <typename U> explicit vec_type(const vec_type<U, N> &o) {
    for (int i = 0; i < N; ++i) data[i] = (T)o.data[i];
  }
};

#define TL_DEFINE_VEC(T)                                                        \
  using T##2 = vec_type<T, 2>;                                                  \
  using T##4 = vec_type<T, 4>;                                                  \
  using T##8 = vec_type<T, 8>;                                                  \
  using T##16 = vec_type<T, 16>;

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
