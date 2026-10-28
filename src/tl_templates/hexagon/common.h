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
#include <type_traits>
#include <utility>

// Hexagon has a native IEEE half type; back the codegen's `half` with it so no
// software emulation is needed on device.
using half = __fp16;

// Kernel status ABI.  A tilelang Hexagon kernel entry returns int32_t — 0 on
// success, nonzero on an unrecoverable device condition.  The FastRPC skel maps
// nonzero to AEE_EFAILED (tl_hexagon_forward_kernel_status in
// runtime/fastrpc.py), so the host's run() raises instead of silently returning
// unwritten/partial output.  The codegen emits these codes into the
// kernel/worker source.
#define TL_OK 0
#define TL_ERR_VTCM                                                            \
  1 // VTCM region unavailable / no worker region fits the grant
#define TL_ERR_HMX 2 // HMX could not be enabled for a worker thread
#define TL_ERR_DMA 3 // User-DMA submission or completion failed
#define TL_ERR_GRID                                                            \
  4 // T.sync_grid: fewer worker threads started than the grid has programs

#ifndef TL_DEVICE
#define TL_DEVICE static inline __attribute__((always_inline))
#endif

// Fixed-width vector vocabulary backing the `floatN`/`halfN`/`intN` spellings
// the C codegen emits (e.g. `*(float4*)(ptr + off)`, `(half4)(x)`, `a * b`).
//
// Backed by a NATIVE clang ext_vector so element-wise ARITHMETIC lowers to real
// HVX ops (vand/vlsr/vmpy/…) instead of scalar loops — a DSL dequant like
// `(int16)(q & 0xF) - 8) * s` becomes native vector `&`/`-`/`*` that
// hexagon-clang++ maps onto HVX.
//
// A vec_type keeps the ext_vector's natural alignment, 128 bytes for a whole
// HVX register, so the codegen's reinterpret loads and stores
// (`*(halfN*)(ptr + off)`) compile to aligned vmem, which ignores the low
// address bits. The codegen prints them only for offsets it proves multiples
// of that alignment, from 128-byte-aligned bases: the VTCM arena's buffers,
// DDR tensors and the stack arrays the codegen aligns. Other vector accesses
// print as tl_hvx_vec_loadu / tl_hvx_vec_storeu (hvx_lanes.h), vmemu.
// `may_alias` makes those accesses legal next to scalar ones on the same array:
// a loop the vectorizer leaves scalar writes `a[i]`, and the next statement may
// read `*(half64*)a`, which strict aliasing would let the compiler reorder.
// The lane type of a comparison of T lanes: a signed integer of T's width,
// all ones where the comparison holds. The codegen prints vector bools as such
// masks (e.g. `a < b`, `!m`, `m && n`) and selects on them (tl_vec_select).
template <int Bytes> struct tl_mask_lane;
template <> struct tl_mask_lane<1> {
  using type = int8_t;
};
template <> struct tl_mask_lane<2> {
  using type = int16_t;
};
template <> struct tl_mask_lane<4> {
  using type = int32_t;
};
template <> struct tl_mask_lane<8> {
  using type = int64_t;
};

// Hides a register or register pair of lanes from the optimizer at no cost.
// LLVM folds a symmetric range check of float lanes (x > -c && x < c), or a
// select that negates the negative ones, into llvm.fabs, which hexagon-clang
// 19 scalarizes for HVX vectors: 128 libcalls for 64 fp16 lanes. A comparison
// of hidden lanes matches neither pattern.
template <typename V> TL_DEVICE V tl_vec_opaque(V v) {
  // "v" is an HVX register, or a register pair for 256 bytes.
  if constexpr (sizeof(V) == 128 || sizeof(V) == 256)
    __asm__("" : "+v"(v));
  return v;
}

// fp32 lanes from integer lanes by the signed conversion instruction, a
// register at a time. LLVM turns the conversion of lanes it proves non-negative
// (a loop index, a zero-extension) into an unsigned one, which hexagon-clang 19
// lowers to IEEE fp adds although -mhvx-ieee-fp is off: the backend fails
// ("V6_vadd_sf_sf"), or the lanes read 0 on the cDSP. Integers of at most 32
// bits but uint32 widen exactly to int32 first.
#if defined(__HVX__) && __HVX_ARCH__ >= 73 && __HVX_LENGTH__ == 128
template <typename T, typename U, int N>
constexpr bool tl_vec_converts_by_vconv =
    std::is_same_v<T, float> && std::is_integral_v<U> && sizeof(U) <= 4 &&
    !std::is_same_v<U, uint32_t> && N % 32 == 0;

template <int N>
inline __attribute__((always_inline)) void tl_vec_vconv_sf_w(void *out,
                                                             const void *in) {
  typedef int words __attribute__((__vector_size__(128)));
#pragma clang loop unroll(full)
  for (int i = 0; i < N / 32; ++i) {
    words x;
    __builtin_memcpy(&x, static_cast<const char *>(in) + 128 * i, 128);
    x = __builtin_HEXAGON_V6_vconv_sf_w_128B(x);
    __builtin_memcpy(static_cast<char *>(out) + 128 * i, &x, 128);
  }
}
#else
template <typename T, typename U, int N>
constexpr bool tl_vec_converts_by_vconv = false;

template <int N> inline void tl_vec_vconv_sf_w(void *, const void *) {}
#endif

template <typename T, int N> struct __attribute__((may_alias)) vec_type {
  typedef T nat_t __attribute__((ext_vector_type(N)));
  using mask_t = vec_type<typename tl_mask_lane<sizeof(T)>::type, N>;
  static constexpr bool is_float =
      std::is_same_v<T, half> || std::is_floating_point_v<T>;
  nat_t v;
  vec_type() = default;
  // Express copies as native vector loads/stores. The implicit aggregate copy
  // becomes llvm.memcpy (and an out-of-line 128-byte memcpy on Hexagon), even
  // for a full-register T.copy between VTCM and DDR.
  vec_type(const vec_type &o) : v(o.v) {}
  vec_type &operator=(const vec_type &o) {
    v = o.v;
    return *this;
  }
  vec_type(nat_t n) : v(n) {} // wrap a native result (operator returns / loads)
  // Broadcast ctor.  const-ref because __fp16 is not a valid by-value parameter
  // type on Hexagon.  `(nat_t)scalar` is clang's ext_vector splat.
  explicit vec_type(const T &s) { v = (nat_t)s; }
  // Converting ctor for vector casts, e.g. (half4)float4_value when storing an
  // fp32 accumulator back as fp16 — a lane-wise numeric convert.
  template <typename U> explicit vec_type(const vec_type<U, N> &o) {
    if constexpr (tl_vec_converts_by_vconv<T, U, N>) {
      typedef int32_t wide_t __attribute__((ext_vector_type(N)));
      wide_t w = __builtin_convertvector(o.v, wide_t);
      tl_vec_vconv_sf_w<N>(&v, &w);
    } else {
      v = __builtin_convertvector(o.v, nat_t);
    }
  }
  // Element-wise operators -> native (HVX) vector ops.  Members are
  // instantiated lazily, so bit-ops on float aggregates (never emitted) never
  // get type-checked.
#define TL_VEC_BINOP(op)                                                       \
  vec_type operator op(const vec_type &o) const { return vec_type(v op o.v); }
  TL_VEC_BINOP(+)
  TL_VEC_BINOP(-)
  TL_VEC_BINOP(*)
  TL_VEC_BINOP(/)
  TL_VEC_BINOP(&)
  TL_VEC_BINOP(|)
  TL_VEC_BINOP(^)
  TL_VEC_BINOP(<<)
  TL_VEC_BINOP(>>)
#undef TL_VEC_BINOP
  vec_type operator-() const { return vec_type(-v); }
  vec_type operator~() const { return vec_type(~v); }
  // Comparisons and logical operators give lane masks. Masks of different
  // widths combine (`m && n` of fp32 and fp16 lanes) at the left one's width.
#define TL_VEC_CMPOP(op)                                                       \
  mask_t operator op(const vec_type &o) const {                                \
    nat_t a = v;                                                               \
    if constexpr (is_float)                                                    \
      a = tl_vec_opaque(a);                                                    \
    return mask_t(__builtin_convertvector(a op o.v, typename mask_t::nat_t));  \
  }
  TL_VEC_CMPOP(==)
  TL_VEC_CMPOP(!=)
  TL_VEC_CMPOP(<)
  TL_VEC_CMPOP(<=)
  TL_VEC_CMPOP(>)
  TL_VEC_CMPOP(>=)
#undef TL_VEC_CMPOP
  mask_t operator!() const {
    return mask_t(__builtin_convertvector(!v, typename mask_t::nat_t));
  }
  template <typename U> mask_t operator&&(const vec_type<U, N> &o) const {
    return mask_t(
        __builtin_convertvector((v != 0) & (o.v != 0), typename mask_t::nat_t));
  }
  template <typename U> mask_t operator||(const vec_type<U, N> &o) const {
    return mask_t(
        __builtin_convertvector((v != 0) | (o.v != 0), typename mask_t::nat_t));
  }
};

// Widths up to 128 lanes: HVX is a 1024-bit vector, so one register holds 128
// int8 / 64 fp16 / 32 fp32.  The codegen's Hexagon vectorizer plans up to that.
#define TL_DEFINE_VEC(T)                                                       \
  using T##2 = vec_type<T, 2>;                                                 \
  using T##4 = vec_type<T, 4>;                                                 \
  using T##8 = vec_type<T, 8>;                                                 \
  using T##16 = vec_type<T, 16>;                                               \
  using T##32 = vec_type<T, 32>;                                               \
  using T##64 = vec_type<T, 64>;                                               \
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

// Lane-wise select, which the codegen prints for a vector Select: a where
// `mask` (lanes of any integer width) is set, b elsewhere. One vcmp and one
// vmux on HVX.
template <typename M, typename T, int N>
TL_DEVICE vec_type<T, N> tl_vec_select(const vec_type<M, N> &mask,
                                       const vec_type<T, N> &a,
                                       const vec_type<T, N> &b) {
  using lanes_t = typename vec_type<T, N>::mask_t::nat_t;
  return vec_type<T, N>(__builtin_convertvector(mask.v, lanes_t) ? a.v : b.v);
}

// The lane indices 0, 1, ..., N - 1: a constant vector.
template <typename V, int... I>
TL_DEVICE V tl_vec_iota(std::integer_sequence<int, I...>) {
  return V{I...};
}

// Lanes base, base + stride, ..., which the codegen prints for a Ramp: a
// splat multiply-add of the constant lane indices.
template <typename T, int N>
TL_DEVICE vec_type<T, N> tl_vec_ramp(const T &base, const T &stride) {
  using nat_t = typename vec_type<T, N>::nat_t;
  return (nat_t)base +
         (nat_t)stride *
             tl_vec_iota<nat_t>(std::make_integer_sequence<int, N>());
}

// Vector min and max, which the codegen prints for vector Min/Max: one HVX
// vmin/vmax for every fp32, fp16 and integer lane type.
template <typename T, int N>
TL_DEVICE vec_type<T, N> tl_vec_min(const vec_type<T, N> &a,
                                    const vec_type<T, N> &b) {
  return __builtin_elementwise_min(a.v, b.v);
}
template <typename T, int N>
TL_DEVICE vec_type<T, N> tl_vec_max(const vec_type<T, N> &a,
                                    const vec_type<T, N> &b) {
  return __builtin_elementwise_max(a.v, b.v);
}
