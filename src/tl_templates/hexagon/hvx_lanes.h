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

// A vector (vec_type) at any address, for the vector accesses the codegen
// cannot prove aligned: vmemu register by register, and for a part of a
// register only the aligned blocks that hold its bytes (defined below).
template <typename V> TL_DEVICE V tl_hvx_vec_loadu(const void *p);
template <typename V> TL_DEVICE void tl_hvx_vec_storeu(void *p, const V &v);

// A whole register at `p`: aligned vmem when the caller knows `p` is on a
// register boundary, else vmemu.
template <bool Aligned> TL_DEVICE HVX_Vector tl_hvx_load(const void *p) {
  if constexpr (Aligned)
    return *(const HVX_Vector *)p;
  else
    return tl_hvx_loadu(p);
}
template <bool Aligned> TL_DEVICE void tl_hvx_store(void *p, HVX_Vector v) {
  if constexpr (Aligned)
    *(HVX_Vector *)p = v;
  else
    tl_hvx_storeu(p, v);
}

// The register-aligned block that holds `p`, and p's offset in it. The block
// address is `p` minus the offset, not `p` with its low bits masked: LLVM then
// keeps the pointer's provenance, so it can tell the block from the stores of
// other buffers, reorder around them and hoist invariant loads.
TL_DEVICE int tl_hvx_block_offset(const void *p) {
  return static_cast<int>(reinterpret_cast<uintptr_t>(p) & 127);
}
TL_DEVICE const HVX_Vector *tl_hvx_block_of(const void *p) {
  return reinterpret_cast<const HVX_Vector *>(static_cast<const char *>(p) -
                                              tl_hvx_block_offset(p));
}

// The `nbytes` (1..128) bytes at `p` in the low bytes of a register; the other
// bytes are undefined. Reads only the aligned blocks that hold those bytes, so
// it never touches memory past them.
TL_DEVICE HVX_Vector tl_hvx_load_bytes(const void *p, int nbytes) {
  int off = tl_hvx_block_offset(p);
  const HVX_Vector *block = tl_hvx_block_of(p);
  HVX_Vector lo = block[0];
  HVX_Vector hi = off + nbytes > 128 ? block[1] : lo;
  return Q6_V_valign_VVR(hi, lo, off);
}

// Stores the low `nbytes` (1..128) bytes of `v` at `p`, leaving the bytes
// around them unchanged.
TL_DEVICE void tl_hvx_store_bytes(void *p, HVX_Vector v, int nbytes) {
  int off = tl_hvx_block_offset(p);
  HVX_Vector *block = const_cast<HVX_Vector *>(tl_hvx_block_of(p));
  // Byte i of v goes to byte off + i of the two blocks.
  HVX_Vector rotated = Q6_V_vror_VR(v, (128 - off) & 127);
  int end = off + nbytes;
  HVX_VectorPred first =
      Q6_Q_and_QQn(Q6_Q_vsetq2_R(end < 128 ? end : 128), Q6_Q_vsetq_R(off));
  Q6_vmem_QRIV(first, block, rotated);
  if (end > 128)
    Q6_vmem_QRIV(Q6_Q_vsetq_R(end - 128), block + 1, rotated);
}

template <typename V> TL_DEVICE V tl_hvx_vec_loadu(const void *p) {
  constexpr int S = static_cast<int>(sizeof(V));
  const char *src = static_cast<const char *>(p);
  V out;
#pragma clang loop unroll(full)
  for (int b = 0; b < S; b += 128) {
    int n = S - b < 128 ? S - b : 128;
    HVX_Vector v =
        n == 128 ? tl_hvx_loadu(src + b) : tl_hvx_load_bytes(src + b, n);
    __builtin_memcpy(reinterpret_cast<char *>(&out) + b, &v, n);
  }
  return out;
}

template <typename V> TL_DEVICE void tl_hvx_vec_storeu(void *p, const V &v) {
  constexpr int S = static_cast<int>(sizeof(V));
  char *dst = static_cast<char *>(p);
#pragma clang loop unroll(full)
  for (int b = 0; b < S; b += 128) {
    int n = S - b < 128 ? S - b : 128;
    HVX_Vector r = Q6_V_vzero();
    __builtin_memcpy(&r, reinterpret_cast<const char *>(&v) + b, n);
    if (n == 128)
      tl_hvx_storeu(dst + b, r);
    else
      tl_hvx_store_bytes(dst + b, r, n);
  }
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

// Lane group i of `x` (at most one register), the G lanes from lane i * G,
// repeated over an N-lane vector: a rotate brings the group to lane 0, and the
// delta network copies it over the register. G = 1 broadcasts lane i.
template <int N, int G, typename T, int M>
TL_DEVICE vec_type<T, N> tl_hvx_broadcast_lane(const vec_type<T, M> &x, int i) {
  static_assert(M * sizeof(T) <= 128, "one register at most");
  constexpr int B = G * static_cast<int>(sizeof(T));
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

// The B bytes at p (a power of two of at most one register) in the low bytes
// of a register. Aligned: p is a multiple of B, so the window lies in one
// aligned block, which one load and a rotate bring down without a branch.
template <int B, bool Aligned>
TL_DEVICE HVX_Vector tl_hvx_load_window(const void *p) {
  static_assert((B & (B - 1)) == 0 && B <= 128, "at most one register");
  if constexpr (Aligned) {
    return Q6_V_vror_VR(*tl_hvx_block_of(p), tl_hvx_block_offset(p));
  } else {
    return tl_hvx_load_bytes(p, B);
  }
}

// An N-lane vector (at most two registers) of the P elements at p, repeated:
// lane l is p[l % P]. The window, P * sizeof(T) bytes, is a power of two of at
// most one register.
template <int P, int N, bool Aligned, typename T>
TL_DEVICE vec_type<T, N> tl_hvx_load_replicate(const T *p) {
  constexpr int B = P * static_cast<int>(sizeof(T));
  constexpr int bytes = N * static_cast<int>(sizeof(T));
  static_assert(bytes <= 256, "two registers at most");
  // Not inside the intrinsic: it is a macro, and the template arguments'
  // comma would split its arguments.
  HVX_Vector window = tl_hvx_load_window<B, Aligned>(p);
  HVX_Vector v = Q6_V_vdelta_VV(window, tl_hvx_replicate_control<B>());
  vec_type<T, N> out;
#pragma clang loop unroll(full)
  for (int b = 0; b < bytes; b += 128)
    __builtin_memcpy(reinterpret_cast<char *>(&out) + b, &v,
                     bytes - b < 128 ? bytes - b : 128);
  return out;
}

// An N-lane vector (at most two registers) of the N / K elements at p, each
// repeated K times: lane l is p[l / K]. Shuffling a register with itself
// repeats each of its chunks twice, so log2(K) shuffles repeat each element
// K times.
template <int K, int N, bool Aligned, typename T>
TL_DEVICE vec_type<T, N> tl_hvx_load_repeat(const T *p) {
  constexpr int B = static_cast<int>(sizeof(T));
  static_assert(K >= 2 && (K & (K - 1)) == 0 && N % K == 0 && N * B <= 256,
                "K a power of two, two registers at most");
  HVX_Vector v = tl_hvx_load_window<N / K * B, Aligned>(p);
  HVX_VectorPair w;
#pragma clang loop unroll(full)
  for (int chunk = B; chunk < K * B; chunk *= 2) {
    w = Q6_W_vshuff_VVR(v, v, -chunk);
    v = Q6_V_lo_W(w);
  }
  vec_type<T, N> out;
  __builtin_memcpy(&out, &w, N * B);
  return out;
}
