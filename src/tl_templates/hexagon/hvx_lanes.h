#pragma once
// Whole HVX registers at any address, parts of registers, and lane broadcasts.
//
// Included by hvx_math.h and hvx_reduce.h.
#include <hexagon_types.h>
#include <stdint.h>
#include <tl_templates/hexagon/common.h>

#ifndef TL_DEVICE
#define TL_DEVICE static inline __attribute__((always_inline))
#endif

// A whole register at any address: `aligned(1)` makes hexagon-clang emit the
// unaligned vmemu form, where a plain HVX_Vector access is an aligned vmem,
// which ignores the low address bits.
typedef HVX_Vector tl_hvx_uvector __attribute__((aligned(1)));
TL_DEVICE HVX_Vector tl_hvx_loadu(const void *p) {
  return *(const tl_hvx_uvector *)p;
}
TL_DEVICE void tl_hvx_storeu(void *p, HVX_Vector v) {
  *(tl_hvx_uvector *)p = v;
}

// A whole register at `p`: aligned vmem when the caller knows `p` is on a
// register boundary, else vmemu.
template <bool Aligned> TL_DEVICE HVX_Vector tl_hvx_load(const void *p) {
  if constexpr (Aligned)
    return *(const HVX_Vector *)p;
  else
    return tl_hvx_loadu(p);
}

// The `nbytes` (1..128) bytes at `p` in the low bytes of a register; the other
// bytes are undefined. Reads only the aligned blocks that hold those bytes, so
// it never touches memory past them.
TL_DEVICE HVX_Vector tl_hvx_load_bytes(const void *p, int nbytes) {
  uintptr_t a = reinterpret_cast<uintptr_t>(p);
  int off = static_cast<int>(a & 127);
  const HVX_Vector *block =
      reinterpret_cast<const HVX_Vector *>(a & ~uintptr_t(127));
  HVX_Vector lo = block[0];
  HVX_Vector hi = off + nbytes > 128 ? block[1] : lo;
  return Q6_V_valign_VVR(hi, lo, off);
}

// Stores the low `nbytes` (1..128) bytes of `v` at `p`, leaving the bytes
// around them unchanged.
TL_DEVICE void tl_hvx_store_bytes(void *p, HVX_Vector v, int nbytes) {
  uintptr_t a = reinterpret_cast<uintptr_t>(p);
  int off = static_cast<int>(a & 127);
  HVX_Vector *block = reinterpret_cast<HVX_Vector *>(a & ~uintptr_t(127));
  // Byte i of v goes to byte off + i of the two blocks.
  HVX_Vector rotated = Q6_V_vror_VR(v, (128 - off) & 127);
  int end = off + nbytes;
  HVX_VectorPred first =
      Q6_Q_and_QQn(Q6_Q_vsetq2_R(end < 128 ? end : 128), Q6_Q_vsetq_R(off));
  Q6_vmem_QRIV(first, block, rotated);
  if (end > 128)
    Q6_vmem_QRIV(Q6_Q_vsetq_R(end - 128), block + 1, rotated);
}

// The `count` (0..N) elements at `p` in the low lanes of an N-lane vector of
// at most one register; the other lanes are undefined.
template <int N, typename T>
TL_DEVICE vec_type<T, N> tl_hvx_load_lanes(const T *p, int count) {
  static_assert(N * sizeof(T) <= 128, "one register at most");
  HVX_Vector v = count > 0
                     ? tl_hvx_load_bytes(p, count * static_cast<int>(sizeof(T)))
                     : Q6_V_vzero();
  vec_type<T, N> out;
  __builtin_memcpy(&out, &v, sizeof(out));
  return out;
}

// The delta-network control that copies the low B bytes of a register over
// all of it: stage k of vdelta takes byte j from byte j ^ 2^k where bit k of
// control byte j is set, so clearing the bits below B copies byte j % B.
template <int B> TL_DEVICE HVX_Vector tl_hvx_replicate_control() {
  alignas(128) static const uint8_t control[128] = {
#define TL_C4(k)                                                               \
  (uint8_t)((k) & ~(B - 1) & 0x7f), (uint8_t)((k + 1) & ~(B - 1) & 0x7f),      \
      (uint8_t)((k + 2) & ~(B - 1) & 0x7f),                                    \
      (uint8_t)((k + 3) & ~(B - 1) & 0x7f)
#define TL_C16(k) TL_C4(k), TL_C4(k + 4), TL_C4(k + 8), TL_C4(k + 12)
#define TL_C64(k) TL_C16(k), TL_C16(k + 16), TL_C16(k + 32), TL_C16(k + 48)
      TL_C64(0), TL_C64(64)
#undef TL_C64
#undef TL_C16
#undef TL_C4
  };
  return *reinterpret_cast<const HVX_Vector *>(control);
}

// Lane i of `x` (at most one register), in every lane of an N-lane vector: a
// rotate brings lane i to lane 0, and the delta network copies it over the
// register.
template <int N, typename T, int M>
TL_DEVICE vec_type<T, N> tl_hvx_broadcast_lane(const vec_type<T, M> &x, int i) {
  static_assert(M * sizeof(T) <= 128, "one register at most");
  constexpr int B = static_cast<int>(sizeof(T));
  HVX_Vector v;
  __builtin_memcpy(&v, &x, sizeof(x));
  v = Q6_V_vdelta_VV(Q6_V_vror_VR(v, i * B), tl_hvx_replicate_control<B>());
  vec_type<T, N> out;
#pragma clang loop unroll(full)
  for (int b = 0; b < N * B; b += 128)
    __builtin_memcpy(reinterpret_cast<char *>(&out) + b, &v,
                     N * B - b < 128 ? N * B - b : 128);
  return out;
}
