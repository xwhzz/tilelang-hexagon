#pragma once
// T.gemm on HVX (tilelang/hexagon/op/gemm/gemm_hvx.py), for the GEMMs HMX does
// not run: M, N or K off multiples of 32, such as decode-time GEMVs, and fp32
// operands.
//
// tl_hvx_gemm<M, N, K, TransA, TransB, LdA, LdB, LdC, AlignA, AlignB,
// AlignC>(C, A, B, clear) computes, for i < M and j < N,
//
//   C[i * LdC + j] = (clear ? 0 : C[i * LdC + j]) + sum_k A(i, k) B(k, j)
//
// where A(i, k) is A[i * LdA + k], or A[k * LdA + i] when TransA, and B(k, j)
// is B[k * LdB + j], or B[j * LdB + k] when TransB; not both transposed. A and
// B are both fp16 or both fp32; C is fp32, or fp16 with fp16 operands. AlignX:
// X starts on a register boundary.
//
// Products accumulate in qf32 registers over the call's K and reach C once.
// v79 has no IEEE fp32 multiply-accumulate (the toolchain emulates the .sf
// forms with qf32 operations, see fastrpc.py), so 64 fp16 multiply-adds are a
// vmpy into a pair of qf32 registers, one of the even lanes and one of the odd
// lanes, and two qf32 adds: four of a packet's four HVX slots, and a fifth for
// the operand's load.
//
// qf32 holds zero in many encodings, and v79's qf32 adds of two zeros step the
// encoding's exponent up until it reads as a huge number. A chain that only
// adds products to one accumulator stays near zero, but two accumulators that
// both hold zero must not meet in a qf32 add: accumulators combine with each
// other, and with C, as IEEE fp32 (sf), whose zero is canonical. A chain of
// zero products converts to the smallest denormals, which the results flush
// to zero, as HMX's do.
//
// Two schedules:
// - Outer products, when B's rows run along N (not TransB): a block of MR rows
//   by NC registers of C accumulates, for each k, A(i, k) broadcast times B's
//   row k, whose registers serve all MR rows. Scalar loads from VTCM are slow,
//   so A's rows reach the broadcasts through a register on the stack, from
//   whose cached lines they are not; a transposed A's MR values for k are one
//   register's lanes, which the delta network broadcasts.
// - Dot products, when A's and B's rows both run along K (TransB): each output
//   accumulates registers of A's row times B's row, two outputs at a time
//   sharing A's loads, into a register of 32 partial sums. A transpose tree
//   sums each output's register into its lane of one register of 32 outputs.
#include <hexagon_types.h>
#include <stdint.h>
#include <tl_templates/hexagon/common.h>
#include <tl_templates/hexagon/hvx_lanes.h>

#include <type_traits>

#ifndef TL_DEVICE
#define TL_DEVICE static inline __attribute__((always_inline))
#endif

constexpr int tl_gemm_min(int a, int b) { return a < b ? a : b; }

constexpr int tl_gemm_log2(int n) {
  return n <= 1 ? 0 : 1 + tl_gemm_log2(n / 2);
}

// The first n elements of Size bytes at p in a register's low lanes: a whole
// register, or only the blocks that hold them (the other lanes undefined).
template <int Size, bool Aligned>
TL_DEVICE HVX_Vector tl_gemm_load(const void *p, int n) {
  if (n * Size == 128)
    return tl_hvx_load<Aligned>(p);
  return tl_hvx_load_bytes(p, n * Size);
}

template <int Size, bool Aligned>
TL_DEVICE void tl_gemm_store(void *p, HVX_Vector v, int n) {
  if (n * Size == 128)
    tl_hvx_store<Aligned>(p, v);
  else
    tl_hvx_store_bytes(p, v, n * Size);
}

// The first n elements of Size bytes at p, zero past them.
template <int Size, bool Aligned>
TL_DEVICE HVX_Vector tl_gemm_load_zeroed(const void *p, int n) {
  HVX_Vector v = tl_gemm_load<Size, Aligned>(p, n);
  if (n * Size == 128)
    return v;
  return Q6_V_vmux_QVV(Q6_Q_vsetq_R(n * Size), v, Q6_V_vzero());
}

TL_DEVICE HVX_Vector tl_gemm_sf(HVX_Vector qf32) {
  return Q6_Vsf_equals_Vqf32(qf32);
}

// a + b of two IEEE fp32 registers, as IEEE fp32.
TL_DEVICE HVX_Vector tl_gemm_add_sf(HVX_Vector a, HVX_Vector b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}

// An IEEE fp32 register with its denormals zero: their exponent field is zero.
// Shifts by scalars find it, so no vector constant stays live across the
// loops that precede it.
TL_DEVICE HVX_Vector tl_gemm_flush(HVX_Vector sf) {
  HVX_Vector exponent = Q6_Vuw_vlsr_VuwR(Q6_Vw_vasl_VwR(sf, 1), 24);
  HVX_VectorPred normal = Q6_Q_vcmp_gt_VuwVuw(exponent, Q6_V_vzero());
  return Q6_V_vmux_QVV(normal, sf, Q6_V_vzero());
}

// An accumulator's qf32 register as IEEE fp32, denormals flushed.
TL_DEVICE HVX_Vector tl_gemm_result(HVX_Vector qf32) {
  return tl_gemm_flush(Q6_Vsf_equals_Vqf32(qf32));
}

// ---------------------------------------------------------------------------
// Operands: a register of A or B, and the accumulator of its products.
// ---------------------------------------------------------------------------

template <typename T> struct tl_gemm_operand;

// fp16: 64 lanes. A register's products are a pair of qf32 registers, the
// even lanes' and the odd lanes'.
template <> struct tl_gemm_operand<half> {
  static constexpr int lanes = 64;
  // Accumulators an outer-product block holds, in pairs of registers: six
  // pairs, its operands and products fit the 32 registers without spills.
  static constexpr int block = 6;
  // An element's bits.
  using bits = uint16_t;
  struct acc {
    HVX_Vector even, odd;
  };
  TL_DEVICE acc zero() { return {Q6_V_vzero(), Q6_V_vzero()}; }
  TL_DEVICE void mac(acc &c, HVX_Vector a, HVX_Vector b) {
    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(a, b);
    c.even = Q6_Vqf32_vadd_Vqf32Vqf32(c.even, Q6_V_lo_W(p));
    c.odd = Q6_Vqf32_vadd_Vqf32Vqf32(c.odd, Q6_V_hi_W(p));
  }
  TL_DEVICE HVX_Vector splat(bits x) { return Q6_Vh_vsplat_R(x); }
  // The register's 32 partial sums of a dot product, as IEEE fp32.
  TL_DEVICE HVX_Vector sums(const acc &c) {
    return tl_gemm_add_sf(tl_gemm_sf(c.even), tl_gemm_sf(c.odd));
  }
};

// fp32: 32 lanes, a register's products one qf32 register.
template <> struct tl_gemm_operand<float> {
  static constexpr int lanes = 32;
  static constexpr int block = 12;
  using bits = int32_t;
  struct acc {
    HVX_Vector v;
  };
  TL_DEVICE acc zero() { return {Q6_V_vzero()}; }
  TL_DEVICE void mac(acc &c, HVX_Vector a, HVX_Vector b) {
    c.v = Q6_Vqf32_vadd_Vqf32Vqf32(c.v, Q6_Vqf32_vmpy_VsfVsf(a, b));
  }
  TL_DEVICE HVX_Vector splat(bits x) { return Q6_V_vsplat_R(x); }
  TL_DEVICE HVX_Vector sums(const acc &c) { return tl_gemm_sf(c.v); }
};

// ---------------------------------------------------------------------------
// C: the n <= lanes columns of a row that one accumulator holds, and the
// n <= 32 outputs of a register of dot products.
// ---------------------------------------------------------------------------

template <typename TA, typename TC> struct tl_gemm_c;

// fp16 operands, fp32 C: the even and odd columns shuffle into two registers
// of 32 columns.
template <> struct tl_gemm_c<half, float> {
  using acc = tl_gemm_operand<half>::acc;
  template <bool Aligned>
  TL_DEVICE void store(float *c, const acc &a, int n, bool clear) {
    HVX_VectorPair s =
        Q6_W_vshuff_VVR(tl_gemm_result(a.odd), tl_gemm_result(a.even), -4);
    HVX_Vector lo = Q6_V_lo_W(s), hi = Q6_V_hi_W(s);
    int n_lo = tl_gemm_min(n, 32);
    if (!clear) {
      HVX_Vector old = tl_gemm_load<4, Aligned>(c, n_lo);
      lo = tl_gemm_add_sf(lo, old);
    }
    tl_gemm_store<4, Aligned>(c, lo, n_lo);
    if (n > 32) {
      if (!clear) {
        HVX_Vector old = tl_gemm_load<4, Aligned>(c + 32, n - 32);
        hi = tl_gemm_add_sf(hi, old);
      }
      tl_gemm_store<4, Aligned>(c + 32, hi, n - 32);
    }
  }
};

// fp16 operands, fp16 C: 64 columns in one register, whose conversion from a
// qf32 pair interleaves the even and odd columns (and to one deals them).
template <> struct tl_gemm_c<half, half> {
  using acc = tl_gemm_operand<half>::acc;
  template <bool Aligned>
  TL_DEVICE void store(half *c, const acc &a, int n, bool clear) {
    HVX_Vector even = tl_gemm_sf(a.even), odd = tl_gemm_sf(a.odd);
    HVX_Vector zero = Q6_V_vzero();
    HVX_Vector old_even = zero, old_odd = zero;
    if (!clear) {
      HVX_Vector old = tl_gemm_load<2, Aligned>(c, n);
      HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(old, Q6_Vh_vsplat_R(0x3c00));
      old_even = tl_gemm_sf(Q6_V_lo_W(w));
      old_odd = tl_gemm_sf(Q6_V_hi_W(w));
    }
    HVX_Vector sum_even = Q6_Vqf32_vadd_VsfVsf(even, old_even);
    HVX_Vector sum_odd = Q6_Vqf32_vadd_VsfVsf(odd, old_odd);
    tl_gemm_store<2, Aligned>(
        c, Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(sum_odd, sum_even)), n);
  }
};

// fp32 operands and C: 32 columns in one register.
template <> struct tl_gemm_c<float, float> {
  using acc = tl_gemm_operand<float>::acc;
  template <bool Aligned>
  TL_DEVICE void store(float *c, const acc &a, int n, bool clear) {
    HVX_Vector v = tl_gemm_result(a.v);
    if (!clear) {
      HVX_Vector old = tl_gemm_load<4, Aligned>(c, n);
      v = tl_gemm_add_sf(v, old);
    }
    tl_gemm_store<4, Aligned>(c, v, n);
  }
};

// The n <= 32 dot products in lanes [0, n) of `sums` (IEEE fp32), written to
// (or, unless clear, added to) c[0, n).
TL_DEVICE void tl_gemm_store_sums(float *c, HVX_Vector sums, int n,
                                  bool clear) {
  sums = tl_gemm_flush(sums);
  if (!clear) {
    HVX_Vector old = tl_gemm_load<4, false>(c, n);
    sums = tl_gemm_add_sf(sums, old);
  }
  tl_gemm_store<4, false>(c, sums, n);
}

TL_DEVICE void tl_gemm_store_sums(half *c, HVX_Vector sums, int n, bool clear) {
  // Outputs 2j and 2j + 1 into lanes j of the even and odd registers, which
  // the conversion to fp16 interleaves.
  HVX_VectorPair d = Q6_W_vdeal_VVR(Q6_V_vzero(), sums, -4);
  HVX_Vector zero = Q6_V_vzero();
  HVX_Vector old_even = zero, old_odd = zero;
  if (!clear) {
    HVX_Vector old = tl_gemm_load<2, false>(c, n);
    HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(old, Q6_Vh_vsplat_R(0x3c00));
    old_even = tl_gemm_sf(Q6_V_lo_W(w));
    old_odd = tl_gemm_sf(Q6_V_hi_W(w));
  }
  HVX_Vector even = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(d), old_even);
  HVX_Vector odd = Q6_Vqf32_vadd_VsfVsf(Q6_V_hi_W(d), old_odd);
  tl_gemm_store<2, false>(c, Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(odd, even)),
                          n);
}

// ---------------------------------------------------------------------------
// Outer products.
// ---------------------------------------------------------------------------

// The block of R rows by Cols columns of C at C, from A's rows (or columns,
// TransA) at A and B's columns at B.
template <typename TA, typename TC, int R, int Cols, int K, bool TransA,
          int LdA, int LdB, int LdC, bool RowsA, bool RowsB, bool RowsC>
TL_DEVICE void tl_gemm_outer(TC *__restrict__ C, const TA *__restrict__ A,
                             const TA *__restrict__ B, bool clear) {
  using V = tl_gemm_operand<TA>;
  using CT = tl_gemm_c<TA, TC>;
  constexpr int L = V::lanes;
  constexpr int S = sizeof(TA);
  constexpr int NC = (Cols + L - 1) / L;
  // Columns of register c: the last one may be partial.
  auto cols = [](int c) { return tl_gemm_min(L, Cols - c * L); };
  typename V::acc acc[R][NC];
#pragma clang loop unroll(full)
  for (int r = 0; r < R; ++r)
#pragma clang loop unroll(full)
    for (int c = 0; c < NC; ++c)
      acc[r][c] = V::zero();
  // Adds A(i, k) B(k, j) for every i, j of the block, A(r, k) broadcast by
  // a(r).
  auto step = [&](int k, auto &&a) {
    HVX_Vector b[NC];
#pragma clang loop unroll(full)
    for (int c = 0; c < NC; ++c)
      b[c] = tl_gemm_load<S, RowsB>(B + k * LdB + c * L, cols(c));
#pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) {
      HVX_Vector ar = a(r);
#pragma clang loop unroll(full)
      for (int c = 0; c < NC; ++c)
        V::mac(acc[r][c], b[c], ar);
    }
  };
  if constexpr (TransA) {
    // A(r, k) for r < R: lanes [0, R) of the register at A's row k.
    const HVX_Vector control = tl_hvx_replicate_control<S>();
    for (int k = 0; k < K; ++k) {
      HVX_Vector v = tl_gemm_load<S, RowsA>(A + k * LdA, R);
      step(k, [&](int r) {
        return Q6_V_vdelta_VV(r == 0 ? v : Q6_V_vror_VR(v, r * S), control);
      });
    }
  } else {
    // A's rows, a register of K at a time, on the stack.
    alignas(128) typename V::bits stage[R][L];
    auto chunk = [&](int k0, int n) {
#pragma clang loop unroll(full)
      for (int r = 0; r < R; ++r) {
        HVX_Vector v = tl_gemm_load<S, RowsA>(A + r * LdA + k0, n);
        __builtin_memcpy(stage[r], &v, sizeof(v));
      }
      for (int kk = 0; kk < n; ++kk)
        step(k0 + kk, [&](int r) { return V::splat(stage[r][kk]); });
    };
#pragma clang loop unroll(disable)
    for (int k0 = 0; k0 + L <= K; k0 += L)
      chunk(k0, L);
    if constexpr (K % L != 0)
      chunk(K - K % L, K % L);
  }
#pragma clang loop unroll(full)
  for (int r = 0; r < R; ++r)
#pragma clang loop unroll(full)
    for (int c = 0; c < NC; ++c)
      CT::template store<RowsC>(C + r * LdC + c * L, acc[r][c], cols(c), clear);
}

// Rows [0, R) of C: blocks of NC registers of columns, the last one narrower.
template <typename TA, typename TC, int R, int N, int K, bool TransA, int LdA,
          int LdB, int LdC, bool RowsA, bool RowsB, bool RowsC, int NC>
TL_DEVICE void tl_gemm_outer_rows(TC *C, const TA *A, const TA *B, bool clear) {
  constexpr int W = NC * tl_gemm_operand<TA>::lanes;
#pragma clang loop unroll(disable)
  for (int j = 0; j + W <= N; j += W)
    tl_gemm_outer<TA, TC, R, W, K, TransA, LdA, LdB, LdC, RowsA, RowsB, RowsC>(
        C + j, A, B + j, clear);
  if constexpr (N % W != 0) {
    constexpr int j = N - N % W;
    tl_gemm_outer<TA, TC, R, N % W, K, TransA, LdA, LdB, LdC, RowsA, RowsB,
                  RowsC>(C + j, A, B + j, clear);
  }
}

template <typename TA, typename TC, int M, int N, int K, bool TransA, int LdA,
          int LdB, int LdC, bool RowsA, bool RowsB, bool RowsC>
TL_DEVICE void tl_gemm_outer_all(TC *C, const TA *A, const TA *B, bool clear) {
  using V = tl_gemm_operand<TA>;
  // Up to four registers of columns, and as many rows as the accumulators
  // left hold: each load of B serves MR rows, each splat of A NC registers.
  constexpr int NC = tl_gemm_min(4, (N + V::lanes - 1) / V::lanes);
  constexpr int MR = tl_gemm_min(M, V::block / NC);
  // A's block of rows i.
  auto a = [A](int i) { return TransA ? A + i : A + i * LdA; };
  // A transposed, its block of rows i starts on a register boundary only for
  // i = 0; the rows of an A that is not do when its row stride is whole
  // registers.
  constexpr bool RowsAi = !TransA && RowsA;
#pragma clang loop unroll(disable)
  for (int i = 0; i + MR <= M; i += MR)
    tl_gemm_outer_rows<TA, TC, MR, N, K, TransA, LdA, LdB, LdC, RowsAi, RowsB,
                       RowsC, NC>(C + i * LdC, a(i), B, clear);
  if constexpr (M % MR != 0) {
    constexpr int i = M - M % MR;
    tl_gemm_outer_rows<TA, TC, M % MR, N, K, TransA, LdA, LdB, LdC, RowsAi,
                       RowsB, RowsC, NC>(C + i * LdC, a(i), B, clear);
  }
}

// ---------------------------------------------------------------------------
// Dot products.
// ---------------------------------------------------------------------------

// Registers holding outputs [0, n) and [n, 2n), each in a run of g lanes, into
// one register holding outputs [0, 2n) in runs of g / 2 lanes (IEEE fp32).
TL_DEVICE HVX_Vector tl_gemm_merge(HVX_Vector lo, HVX_Vector hi) {
  HVX_VectorPair d = Q6_W_vdeal_VVR(hi, lo, -4);
  return tl_gemm_add_sf(Q6_V_lo_W(d), Q6_V_hi_W(d));
}

// The partial sums of outputs (r, g) of rows r < R and the G (1 or 2) rows of
// B at B: for G = 2 merged, output 0 in lanes [0, 16) and output 1 above.
template <typename TA, int R, int G, int K, int LdA, int LdB, bool RowsA,
          bool RowsB>
TL_DEVICE void tl_gemm_dot_leaf(HVX_Vector out[R], const TA *__restrict__ A,
                                const TA *__restrict__ B) {
  using V = tl_gemm_operand<TA>;
  constexpr int L = V::lanes;
  // Registers of K, the last one partial when K is not a multiple of L.
  constexpr int KC = (K + L - 1) / L;
  constexpr int KT = K - (KC - 1) * L;
  constexpr int Whole = KT == L ? KC : KC - 1;
  typename V::acc acc[R][G];
#pragma clang loop unroll(full)
  for (int r = 0; r < R; ++r)
#pragma clang loop unroll(full)
    for (int g = 0; g < G; ++g)
      acc[r][g] = V::zero();
  for (int kc = 0; kc < Whole; ++kc) {
    HVX_Vector b[G];
#pragma clang loop unroll(full)
    for (int g = 0; g < G; ++g)
      b[g] = tl_hvx_load<RowsB>(B + g * LdB + kc * L);
#pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) {
      HVX_Vector a = tl_hvx_load<RowsA>(A + r * LdA + kc * L);
#pragma clang loop unroll(full)
      for (int g = 0; g < G; ++g)
        V::mac(acc[r][g], a, b[g]);
    }
  }
  if constexpr (KT != L) {
    // Both operands zero past K: zero times what lies past a row's end could
    // be a NaN.
    HVX_Vector b[G];
#pragma clang loop unroll(full)
    for (int g = 0; g < G; ++g)
      b[g] =
          tl_gemm_load_zeroed<sizeof(TA), RowsB>(B + g * LdB + Whole * L, KT);
#pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r) {
      HVX_Vector a =
          tl_gemm_load_zeroed<sizeof(TA), RowsA>(A + r * LdA + Whole * L, KT);
#pragma clang loop unroll(full)
      for (int g = 0; g < G; ++g)
        V::mac(acc[r][g], a, b[g]);
    }
  }
#pragma clang loop unroll(full)
  for (int r = 0; r < R; ++r) {
    if constexpr (G == 1)
      out[r] = V::sums(acc[r][0]);
    else
      out[r] = tl_gemm_merge(V::sums(acc[r][0]), V::sums(acc[r][1]));
  }
}

// The outputs [0, Outs), Outs <= 32, of rows r < R of C: pairs of outputs
// enter a transpose tree of P outputs, the smallest power of two that holds
// them, whose pending subtrees wait in `level` by height; then each output's
// run of 32 / P lanes folds into one lane.
template <typename TA, typename TC, int R, int Outs, int K, int LdA, int LdB,
          int LdC, bool RowsA, bool RowsB>
TL_DEVICE void tl_gemm_dot(TC *__restrict__ C, const TA *__restrict__ A,
                           const TA *__restrict__ B, bool clear) {
  constexpr int P = 1 << tl_gemm_log2(2 * Outs - 1);
  constexpr int H = tl_gemm_log2(P);
  HVX_Vector sums[R];
  if constexpr (P == 1) {
    tl_gemm_dot_leaf<TA, R, 1, K, LdA, LdB, RowsA, RowsB>(sums, A, B);
  } else {
    HVX_Vector level[R][H + 1];
    // Adds subtree v of two outputs, the g-th: it merges with the pending
    // subtrees of the heights that g's trailing one bits count.
    auto push = [&level](HVX_Vector *v, int g) {
      int h = 1;
      for (; g & 1; g >>= 1, ++h) {
#pragma clang loop unroll(full)
        for (int r = 0; r < R; ++r)
          v[r] = tl_gemm_merge(level[r][h], v[r]);
      }
#pragma clang loop unroll(full)
      for (int r = 0; r < R; ++r)
        level[r][h] = v[r];
    };
    HVX_Vector v[R];
#pragma clang loop unroll(disable)
    for (int g = 0; g < Outs / 2; ++g) {
      tl_gemm_dot_leaf<TA, R, 2, K, LdA, LdB, RowsA, RowsB>(v, A,
                                                            B + 2 * g * LdB);
      push(v, g);
    }
    if constexpr (Outs % 2 != 0) {
      tl_gemm_dot_leaf<TA, R, 1, K, LdA, LdB, RowsA, RowsB>(
          v, A, B + (Outs - 1) * LdB);
#pragma clang loop unroll(full)
      for (int r = 0; r < R; ++r)
        v[r] = tl_gemm_merge(v[r], Q6_V_vzero());
      push(v, Outs / 2);
    }
#pragma clang loop unroll(disable)
    for (int g = (Outs + 1) / 2; g < P / 2; ++g) {
#pragma clang loop unroll(full)
      for (int r = 0; r < R; ++r)
        v[r] = Q6_V_vzero();
      push(v, g);
    }
#pragma clang loop unroll(full)
    for (int r = 0; r < R; ++r)
      sums[r] = level[r][H];
  }
#pragma clang loop unroll(full)
  for (int r = 0; r < R; ++r) {
#pragma clang loop unroll(full)
    for (int g = 32 / P; g > 1; g /= 2)
      sums[r] = tl_gemm_merge(sums[r], sums[r]);
    tl_gemm_store_sums(C + r * LdC, sums[r], Outs, clear);
  }
}

// Rows [0, R) of C: blocks of 32 outputs, the last one smaller.
template <typename TA, typename TC, int R, int N, int K, int LdA, int LdB,
          int LdC, bool RowsA, bool RowsB>
TL_DEVICE void tl_gemm_dot_rows(TC *C, const TA *A, const TA *B, bool clear) {
#pragma clang loop unroll(disable)
  for (int j = 0; j + 32 <= N; j += 32)
    tl_gemm_dot<TA, TC, R, 32, K, LdA, LdB, LdC, RowsA, RowsB>(
        C + j, A, B + j * LdB, clear);
  if constexpr (N % 32 != 0) {
    constexpr int j = N - N % 32;
    tl_gemm_dot<TA, TC, R, N % 32, K, LdA, LdB, LdC, RowsA, RowsB>(
        C + j, A, B + j * LdB, clear);
  }
}

template <typename TA, typename TC, int M, int N, int K, int LdA, int LdB,
          int LdC, bool RowsA, bool RowsB>
TL_DEVICE void tl_gemm_dot_all(TC *C, const TA *A, const TA *B, bool clear) {
  // Two rows of A share each load of B.
  constexpr int MR = tl_gemm_min(M, 2);
#pragma clang loop unroll(disable)
  for (int i = 0; i + MR <= M; i += MR)
    tl_gemm_dot_rows<TA, TC, MR, N, K, LdA, LdB, LdC, RowsA, RowsB>(
        C + i * LdC, A + i * LdA, B, clear);
  if constexpr (M % MR != 0) {
    constexpr int i = M - M % MR;
    tl_gemm_dot_rows<TA, TC, M % MR, N, K, LdA, LdB, LdC, RowsA, RowsB>(
        C + i * LdC, A + i * LdA, B, clear);
  }
}

template <int M, int N, int K, bool TransA, bool TransB, int LdA, int LdB,
          int LdC, bool AlignA, bool AlignB, bool AlignC, typename TA,
          typename TC>
TL_DEVICE void tl_hvx_gemm(TC *C, const TA *A, const TA *B, bool clear) {
  static_assert(std::is_same_v<TA, half> || std::is_same_v<TA, float>,
                "HVX GEMM operands are fp16 or fp32");
  static_assert(std::is_same_v<TC, float> || std::is_same_v<TA, half>,
                "an fp16 C takes fp16 operands");
  static_assert(!(TransA && TransB),
                "HVX GEMM operands: at most one transposed");
  // Every row of an operand starts on a register boundary.
  constexpr bool RowsA = AlignA && LdA * sizeof(TA) % 128 == 0;
  constexpr bool RowsB = AlignB && LdB * sizeof(TA) % 128 == 0;
  constexpr bool RowsC = AlignC && LdC * sizeof(TC) % 128 == 0;
  if constexpr (TransB)
    tl_gemm_dot_all<TA, TC, M, N, K, LdA, LdB, LdC, RowsA, RowsB>(C, A, B,
                                                                  clear);
  else
    tl_gemm_outer_all<TA, TC, M, N, K, TransA, LdA, LdB, LdC, RowsA, RowsB,
                      RowsC>(C, A, B, clear);
}
