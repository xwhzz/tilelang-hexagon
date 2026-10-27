#pragma once
// Row reductions of the reduce tile op (T.reduce_* along the innermost axis).
//
// Each output row first accumulates its source registers into one register of
// the accumulator type A. A transpose tree then combines the registers of a
// block of rows: vdeal at A's lane size followed by the combine halves the
// columns of two registers and keeps their rows apart, so after log2(lanes)
// levels lane i of the result holds row i. HVX stores the result block; no
// value passes through a scalar register.
//
// Included by hvx_math.h.
#include <hexagon_types.h>
#include <stdint.h>
#include <tl_templates/hexagon/common.h>
#include <tl_templates/hexagon/hvx_lanes.h>

#include <limits>
#include <type_traits>

#ifndef TL_DEVICE
#define TL_DEVICE static inline __attribute__((always_inline))
#endif

// The reduce kinds of the lowering (src/hexagon/op/reduce.cc).
enum {
  TL_REDUCE_SUM = 0,
  TL_REDUCE_MAX = 1,
  TL_REDUCE_MIN = 2,
  TL_REDUCE_ABSSUM = 3,
  TL_REDUCE_ABSMAX = 4,
  TL_REDUCE_AND = 5,
  TL_REDUCE_OR = 6,
  TL_REDUCE_XOR = 7,
};

template <typename T>
constexpr int tl_hvx_lanes = 128 / static_cast<int>(sizeof(T));

// N lanes of T, as a clang vector whose arithmetic lowers to HVX.
template <typename T, int N> struct tl_hvx_lanes_of {
  typedef T type __attribute__((ext_vector_type(N)));
};
template <typename T, int N>
using tl_hvx_vec = typename tl_hvx_lanes_of<T, N>::type;
// One HVX register of T.
template <typename T> using tl_hvx_reg = tl_hvx_vec<T, tl_hvx_lanes<T>>;

template <typename To, typename From>
TL_DEVICE To tl_hvx_bitcast(const From &x) {
  static_assert(sizeof(To) == sizeof(From), "bitcast between equal sizes");
  To y;
  __builtin_memcpy(&y, &x, sizeof(To));
  return y;
}

// ---------------------------------------------------------------------------
// The reduce operations.
// ---------------------------------------------------------------------------
template <int K>
constexpr bool tl_reduce_is_abs =
    K == TL_REDUCE_ABSSUM || K == TL_REDUCE_ABSMAX;

template <typename T>
constexpr bool tl_reduce_is_float =
    std::is_floating_point_v<T> || std::is_same_v<T, half>;

// The combine of two registers, or of two scalars.
template <int K, typename V> TL_DEVICE V tl_reduce_combine(V a, V b) {
  if constexpr (K == TL_REDUCE_SUM || K == TL_REDUCE_ABSSUM)
    return a + b;
  else if constexpr (K == TL_REDUCE_MAX || K == TL_REDUCE_ABSMAX)
    return __builtin_elementwise_max(a, b);
  else if constexpr (K == TL_REDUCE_MIN)
    return __builtin_elementwise_min(a, b);
  else if constexpr (K == TL_REDUCE_AND)
    return a & b;
  else if constexpr (K == TL_REDUCE_OR)
    return a | b;
  else
    return a ^ b;
}

// A register of the identity of the combine in T. (Registers, since __fp16 is
// not a valid return type.)
template <int K, typename T> TL_DEVICE tl_hvx_reg<T> tl_reduce_identity() {
  tl_hvx_reg<T> v;
  if constexpr (K == TL_REDUCE_MAX) {
    if constexpr (tl_reduce_is_float<T>)
      v = static_cast<T>(-INFINITY);
    else
      v = std::numeric_limits<T>::lowest();
  } else if constexpr (K == TL_REDUCE_MIN) {
    if constexpr (tl_reduce_is_float<T>)
      v = static_cast<T>(INFINITY);
    else
      v = std::numeric_limits<T>::max();
  } else if constexpr (K == TL_REDUCE_AND) {
    v = static_cast<T>(~T(0));
  } else {
    // Sums, or and xor; and the abs reductions, whose operands are >= 0.
    v = static_cast<T>(0);
  }
  return v;
}

// A register of source values as operands: absolute values in the abs
// reductions.
template <int K, typename A>
TL_DEVICE tl_hvx_reg<A> tl_reduce_operand(tl_hvx_reg<A> x) {
  if constexpr (!tl_reduce_is_abs<K> || std::is_unsigned_v<A>) {
    return x;
  } else if constexpr (tl_reduce_is_float<A>) {
    // Clear the sign bits.
    const int mask = sizeof(A) == 4 ? 0x7fffffff : 0x7fff7fff;
    return tl_hvx_bitcast<tl_hvx_reg<A>>(
        Q6_V_vand_VV(tl_hvx_bitcast<HVX_Vector>(x), Q6_V_vsplat_R(mask)));
  } else {
    return __builtin_elementwise_abs(x);
  }
}

// ---------------------------------------------------------------------------
// One row: its elements accumulated into one register of A.
// ---------------------------------------------------------------------------

// Source elements per step: one register of S, or as many as fill one
// register of A when S is wider.
template <typename A, typename S>
constexpr int tl_reduce_step =
    tl_hvx_lanes<S> > tl_hvx_lanes<A> ? tl_hvx_lanes<S> : tl_hvx_lanes<A>;
// Source registers per step.
template <typename A, typename S>
constexpr int tl_reduce_step_regs = sizeof(S) > sizeof(A)
                                        ? static_cast<int>(sizeof(S) /
                                                           sizeof(A))
                                        : 1;

// One step of source registers combined into one register of A. A step
// converts to tl_reduce_step<A, S> lanes of A, one or more registers; the
// combine ignores lane order, so fp16 widens with vcvt alone.
template <int K, typename A, typename S>
TL_DEVICE tl_hvx_reg<A>
tl_reduce_step_value(const HVX_Vector (&src)[tl_reduce_step_regs<A, S>]) {
  constexpr int E = tl_reduce_step<A, S>;
  if constexpr (std::is_same_v<A, S>) {
    return tl_reduce_operand<K, A>(tl_hvx_bitcast<tl_hvx_reg<A>>(src[0]));
  } else if constexpr (std::is_same_v<A, float> && std::is_same_v<S, half>) {
    HVX_VectorPair w = Q6_Wsf_vcvt_Vhf(src[0]);
    return tl_reduce_combine<K>(
        tl_reduce_operand<K, A>(tl_hvx_bitcast<tl_hvx_reg<A>>(Q6_V_lo_W(w))),
        tl_reduce_operand<K, A>(tl_hvx_bitcast<tl_hvx_reg<A>>(Q6_V_hi_W(w))));
  } else {
    // Any other pair: a lane-wise conversion of E lanes.
    tl_hvx_vec<S, E> x;
    __builtin_memcpy(&x, src, sizeof(x));
    tl_hvx_vec<A, E> y = __builtin_convertvector(x, tl_hvx_vec<A, E>);
    tl_hvx_reg<A> acc;
    __builtin_memcpy(&acc, &y, 128);
    acc = tl_reduce_operand<K, A>(acc);
#pragma clang loop unroll(full)
    for (int i = 1; i < E / tl_hvx_lanes<A>; ++i) {
      tl_hvx_reg<A> part;
      __builtin_memcpy(&part, reinterpret_cast<const char *>(&y) + 128 * i,
                       128);
      acc = tl_reduce_combine<K>(acc, tl_reduce_operand<K, A>(part));
    }
    return acc;
  }
}

template <int K, typename A, typename S, bool Aligned>
TL_DEVICE tl_hvx_reg<A> tl_reduce_load_step(const S *p) {
  HVX_Vector x[tl_reduce_step_regs<A, S>];
#pragma clang loop unroll(full)
  for (int r = 0; r < tl_reduce_step_regs<A, S>; ++r)
    x[r] = tl_hvx_load<Aligned>(p + r * tl_hvx_lanes<S>);
  return tl_reduce_step_value<K, A, S>(x);
}

// The last `rem` elements of a row, short of a step; the lanes past the row
// take a value that leaves the result unchanged.
template <int K, typename A, typename S>
TL_DEVICE tl_hvx_reg<A> tl_reduce_load_tail(const S *p, int rem) {
  // In the abs reductions, 0 (whose absolute value is the identity).
  tl_hvx_reg<S> pad = tl_reduce_identity<K, S>();
  if constexpr (tl_reduce_is_abs<K>)
    pad = static_cast<S>(0);
  HVX_Vector x[tl_reduce_step_regs<A, S>];
#pragma clang loop unroll(full)
  for (int r = 0; r < tl_reduce_step_regs<A, S>; ++r) {
    int valid = rem - r * tl_hvx_lanes<S>;
    x[r] = tl_hvx_bitcast<HVX_Vector>(pad);
    if (valid > 0) {
      int n = valid < tl_hvx_lanes<S> ? valid : tl_hvx_lanes<S>;
      HVX_Vector v = tl_hvx_load_bytes(p + r * tl_hvx_lanes<S>,
                                       n * static_cast<int>(sizeof(S)));
      x[r] = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(n * static_cast<int>(sizeof(S))), v,
                           x[r]);
    }
  }
  return tl_reduce_step_value<K, A, S>(x);
}

// Reduces the `cols` elements at `row` into one register of A. Long rows keep
// four accumulators, so four combine chains overlap.
template <int K, typename A, typename S, bool Aligned>
TL_DEVICE tl_hvx_reg<A> tl_reduce_row(const S *row, int cols) {
  constexpr int E = tl_reduce_step<A, S>;
  int steps = cols / E;
  int rem = cols - steps * E;
  tl_hvx_reg<A> acc;
  if (steps >= 4) {
    tl_hvx_reg<A> a0 = tl_reduce_load_step<K, A, S, Aligned>(row);
    tl_hvx_reg<A> a1 = tl_reduce_load_step<K, A, S, Aligned>(row + E);
    tl_hvx_reg<A> a2 = tl_reduce_load_step<K, A, S, Aligned>(row + 2 * E);
    tl_hvx_reg<A> a3 = tl_reduce_load_step<K, A, S, Aligned>(row + 3 * E);
    int s = 4;
    for (; s + 4 <= steps; s += 4) {
      a0 = tl_reduce_combine<K>(
          a0, tl_reduce_load_step<K, A, S, Aligned>(row + s * E));
      a1 = tl_reduce_combine<K>(
          a1, tl_reduce_load_step<K, A, S, Aligned>(row + (s + 1) * E));
      a2 = tl_reduce_combine<K>(
          a2, tl_reduce_load_step<K, A, S, Aligned>(row + (s + 2) * E));
      a3 = tl_reduce_combine<K>(
          a3, tl_reduce_load_step<K, A, S, Aligned>(row + (s + 3) * E));
    }
    for (; s < steps; ++s)
      a0 = tl_reduce_combine<K>(
          a0, tl_reduce_load_step<K, A, S, Aligned>(row + s * E));
    acc = tl_reduce_combine<K>(tl_reduce_combine<K>(a0, a1),
                               tl_reduce_combine<K>(a2, a3));
  } else if (steps > 0) {
    acc = tl_reduce_load_step<K, A, S, Aligned>(row);
    for (int s = 1; s < steps; ++s)
      acc = tl_reduce_combine<K>(
          acc, tl_reduce_load_step<K, A, S, Aligned>(row + s * E));
  } else {
    return tl_reduce_load_tail<K, A, S>(row, rem);
  }
  if (rem > 0)
    acc = tl_reduce_combine<K>(
        acc, tl_reduce_load_tail<K, A, S>(row + steps * E, rem));
  return acc;
}

// ---------------------------------------------------------------------------
// The transpose tree.
// ---------------------------------------------------------------------------

// Registers holding rows [0, n) and [n, 2n), each row in a run of g lanes,
// into one register holding rows [0, 2n) in runs of g / 2 lanes.
template <int K, typename A>
TL_DEVICE tl_hvx_reg<A> tl_reduce_merge(tl_hvx_reg<A> lo, tl_hvx_reg<A> hi) {
  HVX_VectorPair d = Q6_W_vdeal_VVR(tl_hvx_bitcast<HVX_Vector>(hi),
                                    tl_hvx_bitcast<HVX_Vector>(lo),
                                    -static_cast<int>(sizeof(A)));
  return tl_reduce_combine<K>(tl_hvx_bitcast<tl_hvx_reg<A>>(Q6_V_lo_W(d)),
                              tl_hvx_bitcast<tl_hvx_reg<A>>(Q6_V_hi_W(d)));
}

// Rows [i0, i0 + N) of a block, N a power of two; rows from n on are padding.
template <int K, typename A, typename S, bool Aligned, int N>
TL_DEVICE tl_hvx_reg<A> tl_reduce_tree(const S *src, int src_stride, int cols,
                                       int i0, int n) {
  if constexpr (N == 1) {
    if (i0 >= n)
      return tl_reduce_identity<K, A>();
    return tl_reduce_row<K, A, S, Aligned>(src + i0 * src_stride, cols);
  } else {
    return tl_reduce_merge<K, A>(
        tl_reduce_tree<K, A, S, Aligned, N / 2>(src, src_stride, cols, i0, n),
        tl_reduce_tree<K, A, S, Aligned, N / 2>(src, src_stride, cols,
                                                i0 + N / 2, n));
  }
}

// A tree over the registers part[i0, i0 + M), M a power of two.
template <int K, typename A, int M>
TL_DEVICE tl_hvx_reg<A> tl_reduce_merge_all(const tl_hvx_reg<A> *part, int i0) {
  if constexpr (M == 1)
    return part[i0];
  else
    return tl_reduce_merge<K, A>(
        tl_reduce_merge_all<K, A, M / 2>(part, i0),
        tl_reduce_merge_all<K, A, M / 2>(part, i0 + M / 2));
}

// Rows [0, N) of a block, N a power of two, rows from n on padding. Rows of
// one or two whole steps all inline into one tree. Other rows run in a loop
// over groups of four, whose trees meet in a tree of the groups, so the code
// holds four row bodies whatever N is: 64 inlined rows with tails spill to a
// 17 KB stack frame, past what a FastRPC thread's stack holds.
template <int K, typename A, typename S, bool Aligned, int N>
TL_DEVICE tl_hvx_reg<A> tl_reduce_rows_of(const S *src, int src_stride,
                                          int cols, int n) {
  constexpr int G = 4;
  constexpr int E = tl_reduce_step<A, S>;
  if constexpr (N > G) {
    if (cols % E != 0 || cols > 2 * E) {
      tl_hvx_reg<A> part[N / G];
#pragma clang loop unroll(disable)
      for (int g = 0; g < N / G; ++g)
        part[g] = tl_reduce_tree<K, A, S, Aligned, G>(
            src + g * G * src_stride, src_stride, cols, 0, n - g * G);
      return tl_reduce_merge_all<K, A, N / G>(part, 0);
    }
  }
  return tl_reduce_tree<K, A, S, Aligned, N>(src, src_stride, cols, 0, n);
}

// The results of rows [0, n), 0 < n <= N, in lanes [0, n): a tree over the
// smallest power of two that holds n rows, then folds of each row's run.
template <int K, typename A, typename S, bool Aligned, int N>
TL_DEVICE tl_hvx_reg<A> tl_reduce_block(const S *src, int src_stride, int cols,
                                        int n) {
  if constexpr (N > 1) {
    if (n <= N / 2)
      return tl_reduce_block<K, A, S, Aligned, N / 2>(src, src_stride, cols, n);
  }
  tl_hvx_reg<A> v =
      tl_reduce_rows_of<K, A, S, Aligned, N>(src, src_stride, cols, n);
#pragma clang loop unroll(full)
  for (int g = tl_hvx_lanes<A> / N; g > 1; g /= 2)
    v = tl_reduce_merge<K, A>(v, v);
  return v;
}

// Writes the results of n rows, lanes [0, n) of `v`, to dst[i * dst_stride],
// combined with the values already there unless `clear`.
template <int K, typename A, typename D>
TL_DEVICE void tl_reduce_store(D *dst, int dst_stride, tl_hvx_reg<A> v, int n,
                               bool clear) {
  constexpr int L = tl_hvx_lanes<A>;
  if (dst_stride != 1) {
    // Scalars combine in fp32 when A is fp16 (exact for max and min).
    using C = std::conditional_t<std::is_same_v<A, half>, float, A>;
    A lanes[L] __attribute__((aligned(128)));
    __builtin_memcpy(lanes, &v, 128);
    for (int i = 0; i < n; ++i) {
      C x = static_cast<C>(lanes[i]);
      if (!clear)
        x = tl_reduce_combine<K>(static_cast<C>(dst[i * dst_stride]), x);
      dst[i * dst_stride] = static_cast<D>(x);
    }
    return;
  }
  // L lanes of D: one register, half of one, or two.
  typedef tl_hvx_vec<D, L> out_t;
  constexpr int out_bytes = static_cast<int>(sizeof(out_t));
  const int bytes = n * static_cast<int>(sizeof(D));
  if (!clear) {
    out_t old;
#pragma clang loop unroll(full)
    for (int b = 0; b < out_bytes; b += 128) {
      if (bytes > b) {
        HVX_Vector part =
            tl_hvx_load_bytes(reinterpret_cast<const char *>(dst) + b,
                              bytes - b < 128 ? bytes - b : 128);
        __builtin_memcpy(reinterpret_cast<char *>(&old) + b, &part,
                         out_bytes - b < 128 ? out_bytes - b : 128);
      }
    }
    v = tl_reduce_combine<K>(v, __builtin_convertvector(old, tl_hvx_reg<A>));
  }
  out_t out = __builtin_convertvector(v, out_t);
#pragma clang loop unroll(full)
  for (int b = 0; b < out_bytes; b += 128) {
    if (bytes > b) {
      HVX_Vector part;
      __builtin_memcpy(&part, reinterpret_cast<const char *>(&out) + b,
                       out_bytes - b < 128 ? out_bytes - b : 128);
      tl_hvx_store_bytes(reinterpret_cast<char *>(dst) + b, part,
                         bytes - b < 128 ? bytes - b : 128);
    }
  }
}

// Reduces `rows` rows of `cols` elements, src[r * src_stride + c], into
// dst[r * dst_stride], accumulating in A. Combines with dst unless `clear`.
// Aligned: every row starts on a 128-byte boundary.
template <int K, typename A, bool Aligned, typename D, typename S>
TL_DEVICE void tl_hvx_reduce_rows(D *dst, const S *src, int rows, int cols,
                                  int src_stride, int dst_stride, bool clear) {
  constexpr int L = tl_hvx_lanes<A>;
  int blocks = rows / L;
  for (int b = 0; b < blocks; ++b) {
    tl_hvx_reg<A> v = tl_reduce_rows_of<K, A, S, Aligned, L>(
        src + b * L * src_stride, src_stride, cols, L);
    tl_reduce_store<K, A>(dst + b * L * dst_stride, dst_stride, v, L, clear);
  }
  int n = rows - blocks * L;
  if (n > 0) {
    tl_hvx_reg<A> v = tl_reduce_block<K, A, S, Aligned, L>(
        src + blocks * L * src_stride, src_stride, cols, n);
    tl_reduce_store<K, A>(dst + blocks * L * dst_stride, dst_stride, v, n,
                          clear);
  }
}
