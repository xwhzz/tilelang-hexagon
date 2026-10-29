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

// A GEMM is a function of its own, never inlined into its caller: the calls of
// one shape share one copy of its code, so a kernel that multiplies in many
// places, as a megakernel's phases do, stays within the instruction cache.
#define TL_GEMM static __attribute__((noinline))

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
TL_GEMM void tl_hvx_gemm(TC *C, const TA *A, const TA *B, bool clear) {
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

// ---------------------------------------------------------------------------
// B as an HMX weight operand: Crouton tiles (src/hexagon/layout/crouton.h).
// ---------------------------------------------------------------------------
//
// tl_hvx_gemm_weight<M, N, K, LdA, LdC, AlignA, PanelStride>(C, A, B, clear)
// computes, for i < M and j < N,
//
//   C[i * LdC + j] (+)= sum_k A[i * LdA + k] B(k, j)
//
// (adding unless clear) for an fp16 A and the fp16 B [K, N] of the HMX weight
// operand: 32 x 32 Crouton tiles ordered by columns, the K / 32 tiles of each
// panel of 32 columns contiguous, panel p at B + p * PanelStride, on a register
// boundary. C is fp32 or fp16. A decode's GEMVs read the copy of the weights
// that HMX multiplies in the prefill.
//
// Register r of a tile holds its rows 2r and 2r + 1 interleaved: lane 2j row
// 2r of column j, lane 2j + 1 row 2r + 1. A's elements 2r and 2r + 1 of the
// tile's K, one 32-bit word broadcast, multiply it into the even row's
// products and the odd row's (tl_gemm_operand<half>::mac), each in its
// column's lane: over a panel's tiles, the two sum to its 32 outputs, in
// order, with no reduction across lanes.

// Tiles [T0, T0 + NT) of K, NT of 1 or 2 (one register of A), of G panels of
// B into their accumulators: U of them a panel, register r of a tile into
// accumulator r % U, so that a call of one or two panels still spreads its
// qf32 adds over eight chains.
template <int G, int U, int NT, int PanelStride, bool AlignA>
TL_DEVICE void tl_gemm_weight_tiles(tl_gemm_operand<half>::acc acc[G * U],
                                    const half *__restrict__ a,
                                    const half *__restrict__ B, int t0) {
  using V = tl_gemm_operand<half>;
  // A's pairs reach the broadcasts through the stack: scalar loads from VTCM
  // are slow.
  alignas(128) int32_t pairs[32];
  HVX_Vector v = tl_gemm_load<2, AlignA>(a + t0 * 32, NT * 32);
  __builtin_memcpy(pairs, &v, sizeof(v));
#pragma clang loop unroll(full)
  for (int t = 0; t < NT; ++t) {
    const half *tile = B + (t0 + t) * 1024;
#pragma clang loop unroll(full)
    for (int r = 0; r < 16; ++r) {
      HVX_Vector x = Q6_V_vsplat_R(pairs[t * 16 + r]);
#pragma clang loop unroll(full)
      for (int p = 0; p < G; ++p)
        V::mac(acc[p * U + r % U],
               *(const HVX_Vector *)(tile + p * PanelStride + r * 64), x);
    }
  }
}

// Panels [0, G) of B times A's row a, each panel's 32 outputs into c.
template <int G, int K, int PanelStride, bool AlignA, typename TC>
TL_DEVICE void tl_gemm_weight_panels(TC *__restrict__ c,
                                     const half *__restrict__ a,
                                     const half *__restrict__ B, bool clear) {
  using V = tl_gemm_operand<half>;
  constexpr int KT = K / 32;
  constexpr int U = 4 / G;
  typename V::acc acc[G * U];
#pragma clang loop unroll(full)
  for (int i = 0; i < G * U; ++i)
    acc[i] = V::zero();
#pragma clang loop unroll(disable)
  for (int t = 0; t + 2 <= KT; t += 2)
    tl_gemm_weight_tiles<G, U, 2, PanelStride, AlignA>(acc, a, B, t);
  if constexpr (KT % 2 != 0)
    tl_gemm_weight_tiles<G, U, 1, PanelStride, AlignA>(acc, a, B, KT - 1);
  // A panel's accumulators meet as IEEE fp32: two qf32 zeros must not.
#pragma clang loop unroll(full)
  for (int p = 0; p < G; ++p) {
    HVX_Vector sums = V::sums(acc[p * U]);
#pragma clang loop unroll(full)
    for (int u = 1; u < U; ++u)
      sums = tl_gemm_add_sf(sums, V::sums(acc[p * U + u]));
    tl_gemm_store_sums(c + p * 32, sums, 32, clear);
  }
}

template <int M, int N, int K, int LdA, int LdC, bool AlignA, int PanelStride,
          typename TC>
TL_GEMM void tl_hvx_gemm_weight(TC *C, const half *A, const half *B,
                                bool clear) {
  static_assert(N % 32 == 0 && K % 32 == 0,
                "a weight operand covers whole Crouton tiles");
  // Four panels share each broadcast of A: their accumulators, loads and
  // products fit the registers.
  constexpr int P = N / 32;
  constexpr int G = tl_gemm_min(4, P);
  constexpr bool RowsA = AlignA && LdA * sizeof(half) % 128 == 0;
#pragma clang loop unroll(disable)
  for (int i = 0; i < M; ++i) {
    const half *a = A + i * LdA;
    TC *c = C + i * LdC;
#pragma clang loop unroll(disable)
    for (int p = 0; p + G <= P; p += G)
      tl_gemm_weight_panels<G, K, PanelStride, RowsA>(
          c + p * 32, a, B + p * PanelStride, clear);
    if constexpr (P % G != 0) {
      constexpr int p = P - P % G;
      tl_gemm_weight_panels<P % G, K, PanelStride, RowsA>(
          c + p * 32, a, B + p * PanelStride, clear);
    }
  }
}

// ---------------------------------------------------------------------------
// Integer GEMM: B in the panel layout (src/hexagon/layout/panel.h).
// ---------------------------------------------------------------------------
//
// tl_hvx_gemm_int<M, N, K, BBits, BSigned, LdA, LdC, AlignC>(C, A, B,
// panel_bytes, clear) computes, for i < M and j < N, exactly in int32,
//
//   C[i * LdC + j] = (clear ? 0 : C[i * LdC + j]) + sum_k A[i * LdA + k] B(j,
//   k)
//
// for an int8, uint8 or int16 A and a BBits-bit B (8 or 4), signed or not,
// whose panel of rows [32 p, 32 p + 32) starts p * panel_bytes bytes into B.
//
// vrmpy multiplies the 4 bytes of each 32-bit lane of a register by the 4
// bytes of a scalar, and adds the products into the lane of an int32
// accumulator. A register of B is a column block of a panel: 32 rows, so 32
// outputs, and in each row's lane the next 4 (8-bit) or 8 (4-bit) k.
// - The scalars hold A's values at those k. A's row reaches them through the
//   stack, its bytes regrouped into 32-bit words by the k a lane holds: two
//   4-bit columns share a byte, so a register's low nibbles hold the even k
//   of each lane and its high nibbles the odd k.
// - An int16 a is 256 * hi + lo, hi its signed and lo its unsigned byte: two
//   vrmpy per register, into two accumulators that meet at the end.
// - vrmpy reads B's bytes unsigned: a signed b with its sign bit flipped is
//   b + 128 (8-bit) or b + 8 (4-bit), and the call subtracts that offset
//   times the sum of A's row.

// A's row in 32-bit words for vrmpy, at most KC columns: per 128 columns and
// per byte plane of A (hi then lo for int16), one register of words; with
// 4-bit B its low half holds the even columns' words, its high half the odd
// columns'.
template <typename TA, int BBits, int KC> struct tl_gemm_int_row {
  static constexpr int planes = sizeof(TA);
  static constexpr int chunks = (KC + 127) / 128;
  alignas(128) int32_t words[chunks][planes][32];
  // The row's sum, for a signed B.
  int32_t sum = 0;

  TL_DEVICE HVX_Vector deal(HVX_Vector v) {
    if constexpr (BBits == 4)
      return Q6_Vb_vdeal_Vb(v);
    return v;
  }

  inline __attribute__((always_inline)) void put(int c, int plane,
                                                 HVX_Vector v) {
    v = deal(v);
    __builtin_memcpy(words[c][plane], &v, sizeof(v));
  }

  // Columns [0, k) of the row at `a`, k <= KC; past k, zeros.
  inline __attribute__((always_inline)) void stage(const TA *a, int k,
                                                   bool want_sum) {
    HVX_Vector sums = Q6_V_vzero();
#pragma clang loop unroll(disable)
    for (int c = 0; c * 128 < k; ++c) {
      int n = tl_gemm_min(128, k - c * 128);
      if constexpr (planes == 2) {
        HVX_Vector v0 =
            tl_gemm_load_zeroed<2, false>(a + c * 128, tl_gemm_min(n, 64));
        HVX_Vector v1 =
            n > 64 ? tl_gemm_load_zeroed<2, false>(a + c * 128 + 64, n - 64)
                   : Q6_V_vzero();
        put(c, 0, Q6_Vb_vpacko_VhVh(v1, v0));
        put(c, 1, Q6_Vb_vpacke_VhVh(v1, v0));
        if (want_sum) {
          sums = Q6_Vw_vadd_VwVw(sums, Q6_Vw_vdmpy_VhRh_sat(v0, 0x00010001));
          sums = Q6_Vw_vadd_VwVw(sums, Q6_Vw_vdmpy_VhRh_sat(v1, 0x00010001));
        }
      } else {
        HVX_Vector v = tl_gemm_load_zeroed<1, false>(a + c * 128, n);
        put(c, 0, v);
        if (want_sum) {
          if constexpr (std::is_signed_v<TA>)
            sums = Q6_Vw_vrmpyacc_VwVbVb(sums, v, Q6_Vb_vsplat_R(1));
          else
            sums = Q6_Vuw_vrmpyacc_VuwVubRub(sums, v, 0x01010101);
        }
      }
    }
    if (want_sum) {
      alignas(128) int32_t lanes[32];
      __builtin_memcpy(lanes, &sums, sizeof(sums));
      int32_t total = 0;
      for (int i = 0; i < 32; ++i)
        total += lanes[i];
      sum = total;
    }
  }

  // The word of `plane` for the columns of step `s`, of the lane's 4 (8-bit
  // B) or 8 (4-bit) columns; `odd` picks the odd columns of a 4-bit step.
  inline __attribute__((always_inline)) int32_t word(int plane, int s,
                                                     bool odd) const {
    if constexpr (BBits == 4)
      return words[s / 16][plane][(odd ? 16 : 0) + s % 16];
    return words[s / 32][plane][s % 32];
  }
};

// acc + the products of each lane's 4 unsigned bytes of u with the 4 bytes
// of w, signed or unsigned.
template <bool Signed>
TL_DEVICE HVX_Vector tl_gemm_vrmpy(HVX_Vector acc, HVX_Vector u, int32_t w) {
  if constexpr (Signed)
    return Q6_Vw_vrmpyacc_VwVubRb(acc, u, w);
  return Q6_Vuw_vrmpyacc_VuwVubRub(acc, u, w);
}

// Accumulates the panels [p, p + NB) of B for the staged row `row` over its
// `steps` column blocks, then writes the row's 32 * NB outputs at C.
template <typename TA, int BBits, bool BSigned, int NB, int KC, bool AlignC>
TL_DEVICE void
tl_gemm_int_panels(int32_t *C, const tl_gemm_int_row<TA, BBits, KC> &row,
                   const uint8_t *B, int panel_bytes, int steps, bool clear) {
  constexpr int planes = sizeof(TA);
  HVX_Vector acc[planes][NB];
#pragma clang loop unroll(full)
  for (int q = 0; q < planes; ++q)
#pragma clang loop unroll(full)
    for (int b = 0; b < NB; ++b)
      acc[q][b] = Q6_V_vzero();
  const HVX_Vector nibble = Q6_Vb_vsplat_R(0x0F);
  const HVX_Vector flip = Q6_Vb_vsplat_R(BBits == 4 ? 0x88 : 0x80);
  // The scalar bytes of int16's hi plane and of int8 are signed.
  constexpr bool hi_signed = planes == 2 || std::is_signed_v<TA>;
#pragma clang loop unroll_count(2)
  for (int s = 0; s < steps; ++s) {
#pragma clang loop unroll(full)
    for (int b = 0; b < NB; ++b) {
      HVX_Vector v =
          *reinterpret_cast<const HVX_Vector *>(B + b * panel_bytes + s * 128);
      if constexpr (BSigned)
        v = Q6_V_vxor_VV(v, flip);
      HVX_Vector u[2] = {v, v};
      if constexpr (BBits == 4) {
        u[0] = Q6_V_vand_VV(v, nibble);
        u[1] = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v, 4), nibble);
      }
#pragma clang loop unroll(full)
      for (int h = 0; h < (BBits == 4 ? 2 : 1); ++h) {
        acc[0][b] =
            tl_gemm_vrmpy<hi_signed>(acc[0][b], u[h], row.word(0, s, h == 1));
        if constexpr (planes == 2)
          acc[1][b] =
              tl_gemm_vrmpy<false>(acc[1][b], u[h], row.word(1, s, h == 1));
      }
    }
  }
  // The offset of a signed B's bytes times the row's sum.
  const int32_t offset = BSigned ? (BBits == 4 ? 8 : 128) * row.sum : 0;
#pragma clang loop unroll(full)
  for (int b = 0; b < NB; ++b) {
    HVX_Vector r = acc[0][b];
    if constexpr (planes == 2)
      r = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(r, 8), acc[1][b]);
    if constexpr (BSigned)
      r = Q6_Vw_vsub_VwVw(r, Q6_V_vsplat_R(offset));
    if (!clear)
      r = Q6_Vw_vadd_VwVw(r, tl_hvx_load<AlignC>(C + 32 * b));
    tl_hvx_store<AlignC>(C + 32 * b, r);
  }
}

template <int M, int N, int K, int BBits, bool BSigned, int LdA, int LdC,
          bool AlignC, typename TA>
TL_GEMM void tl_hvx_gemm_int(int32_t *C, const TA *A, const void *B_,
                             int panel_bytes, bool clear) {
  const uint8_t *B = static_cast<const uint8_t *>(B_);
  static_assert(std::is_same_v<TA, int8_t> || std::is_same_v<TA, uint8_t> ||
                    std::is_same_v<TA, int16_t>,
                "HVX integer GEMM: A is int8, uint8 or int16");
  static_assert(BBits == 4 || BBits == 8, "HVX integer GEMM: B is 4 or 8 bits");
  constexpr int lane = 32 / BBits;
  static_assert(N % 32 == 0 && K % lane == 0,
                "HVX integer GEMM: B covers whole panels");
  // Columns per staged chunk of A's row, and panels per accumulator block.
  constexpr int KC = tl_gemm_min(K, 512);
  constexpr int NB = 4;
  constexpr bool RowsC = AlignC && LdC % 32 == 0;
  tl_gemm_int_row<TA, BBits, KC> row;
#pragma clang loop unroll(disable)
  for (int i = 0; i < M; ++i) {
#pragma clang loop unroll(disable)
    for (int k0 = 0; k0 < K; k0 += KC) {
      int k = tl_gemm_min(KC, K - k0);
      row.stage(A + i * LdA + k0, k, BSigned);
      bool first = clear && k0 == 0;
      const uint8_t *b0 = B + k0 / lane * 128;
      int32_t *c0 = C + i * LdC;
#pragma clang loop unroll(disable)
      for (int p = 0; p + NB <= N / 32; p += NB)
        tl_gemm_int_panels<TA, BBits, BSigned, NB, KC, RowsC>(
            c0 + 32 * p, row, b0 + p * panel_bytes, panel_bytes, k / lane,
            first);
      if constexpr (N / 32 % NB != 0) {
        constexpr int p = N / 32 - N / 32 % NB;
        tl_gemm_int_panels<TA, BBits, BSigned, N / 32 % NB, KC, RowsC>(
            c0 + 32 * p, row, b0 + p * panel_bytes, panel_bytes, k / lane,
            first);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Block-scaled integer GEMM (T.gemm_blockscaled): C += (A * SFA) (B * SFB)^T.
// ---------------------------------------------------------------------------
//
// tl_hvx_gemm_int_scaled<M, N, K, BBits, BSigned, LdA, LdC, GA, GB,
// AlignC>(C, A, B, panel_bytes, sfa, ld_sfa, sfb, ld_sfb, clear) computes,
// for i < M and j < N,
//
//   C[i * LdC + j] = (clear ? 0 : C[i * LdC + j]) +
//       sum_g sfa[i * ld_sfa + g / (GA / G)] sfb[(g / (GB / G)) * ld_sfb + j]
//             dot_g(i, j)
//
// over the groups g of G = min(GA, GB) columns of K, with dot_g(i, j) the
// exact int32 sum over group g of A[i, k] B(j, k): B as in tl_hvx_gemm_int,
// the scales fp32 (SFA, one per GA columns) and fp16 (SFB, one per GB columns,
// group-major). GA and GB divide K, or equal it for one factor per row. Each
// group's int32 sums leave the vrmpy accumulators once, into fp32 accumulators
// that weigh them by the group's scales: the scaling costs a few operations
// per group of 32 outputs, against a vrmpy for every 4 or 8 columns of them.

// The fp32 values of NB runs of 32 consecutive fp16 from p: two runs per
// conversion, its pair of even and odd lanes shuffled back in order. Scales
// are never denormal, so they skip the flush.
template <int NB>
TL_DEVICE void tl_gemm_half_rows_sf(HVX_Vector out[NB], const half *p) {
  const HVX_Vector one = Q6_Vh_vsplat_R(0x3C00);
#pragma clang loop unroll(full)
  for (int b = 0; b < NB; b += 2) {
    HVX_Vector h = b + 1 < NB ? tl_hvx_load<false>(p + 32 * b)
                              : tl_hvx_load_bytes(p + 32 * b, 64);
    HVX_VectorPair q = Q6_Wqf32_vmpy_VhfVhf(h, one);
    HVX_VectorPair ordered =
        Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(q)),
                        Q6_Vsf_equals_Vqf32(Q6_V_lo_W(q)), -4);
    out[b] = Q6_V_lo_W(ordered);
    if (b + 1 < NB)
      out[b + 1] = Q6_V_hi_W(ordered);
  }
}

// Two int32 registers summing 2n values in runs of g lanes each, [0, n) and
// [n, 2n), into one summing them in runs of g / 2 lanes (tl_gemm_merge's int32
// form).
TL_DEVICE HVX_Vector tl_gemm_int_merge(HVX_Vector lo, HVX_Vector hi) {
  HVX_VectorPair d = Q6_W_vdeal_VVR(hi, lo, -4);
  return Q6_Vw_vadd_VwVw(Q6_V_lo_W(d), Q6_V_hi_W(d));
}

// The sum of each group of Group columns of an int16 row of K columns, for a
// signed B's offset: each group's registers summed in pairs of lanes (vdmpy),
// 32 groups at a time through a transpose tree that holds only its pending
// subtrees, one per height.
template <int K, int Group> struct tl_gemm_int_group_sums {
  static_assert(Group % 64 == 0,
                "a signed B's groups take whole registers of an int16 A");
  static constexpr int groups = K / Group;
  alignas(128) int32_t sum[(groups + 31) / 32 * 32];

  inline __attribute__((always_inline)) void compute(const int16_t *a) {
#pragma clang loop unroll(disable)
    for (int g0 = 0; g0 < groups; g0 += 32) {
      HVX_Vector level[6];
#pragma clang loop unroll(full)
      for (int g = 0; g < 32; ++g) {
        HVX_Vector v = Q6_V_vzero();
        if (g0 + g < groups) {
#pragma clang loop unroll(full)
          for (int c = 0; c < Group; c += 64)
            v = Q6_Vw_vadd_VwVw(
                v,
                Q6_Vw_vdmpy_VhRh_sat(
                    tl_hvx_load<false>(a + (g0 + g) * Group + c), 0x00010001));
        }
        // Merges with the pending subtrees of the heights g's trailing one
        // bits count.
        int h = 0;
        for (int n = g; n & 1; n >>= 1, ++h)
          v = tl_gemm_int_merge(level[h], v);
        level[h] = v;
      }
      __builtin_memcpy(sum + g0, &level[5], sizeof(HVX_Vector));
    }
  }
};

// The groups [g0, g0 + groups) of NB panels, from the staged chunk of A's row
// that starts at group g0, added to their fp32 sums `total`. sfa and sfb hold
// the factors of the row of A and of the panels' rows of B, from group 0.
template <typename TA, int BBits, bool BSigned, int NB, int KC, int Group,
          bool GroupA>
TL_DEVICE void tl_gemm_int_panels_scaled(
    HVX_Vector *total, const tl_gemm_int_row<TA, BBits, KC> &row,
    const int32_t *sum, const uint8_t *B, int panel_bytes, int g0, int groups,
    const float *sfa, int sfa_per, const half *sfb, int ld_sfb, int sfb_per) {
  constexpr int planes = sizeof(TA);
  constexpr int lane = 32 / BBits;
  constexpr int steps = Group / lane;
  constexpr bool hi_signed = planes == 2 || std::is_signed_v<TA>;
  const HVX_Vector nibble = Q6_Vb_vsplat_R(0x0F);
  const HVX_Vector flip = Q6_Vb_vsplat_R(BBits == 4 ? 0x88 : 0x80);
#pragma clang loop unroll(disable)
  for (int g = 0; g < groups; ++g) {
    HVX_Vector acc[planes][NB];
#pragma clang loop unroll(full)
    for (int q = 0; q < planes; ++q)
#pragma clang loop unroll(full)
      for (int b = 0; b < NB; ++b)
        acc[q][b] = Q6_V_vzero();
// Partly unrolled: a whole group of steps would spill the registers it
// keeps live onto the stack.
#pragma clang loop unroll_count(2)
    for (int t = 0; t < steps; ++t) {
      int s = g * steps + t;
#pragma clang loop unroll(full)
      for (int b = 0; b < NB; ++b) {
        HVX_Vector v = *reinterpret_cast<const HVX_Vector *>(
            B + b * panel_bytes + s * 128);
        if constexpr (BSigned)
          v = Q6_V_vxor_VV(v, flip);
        HVX_Vector u[2] = {v, v};
        if constexpr (BBits == 4) {
          u[0] = Q6_V_vand_VV(v, nibble);
          u[1] = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v, 4), nibble);
        }
#pragma clang loop unroll(full)
        for (int h = 0; h < (BBits == 4 ? 2 : 1); ++h) {
          acc[0][b] =
              tl_gemm_vrmpy<hi_signed>(acc[0][b], u[h], row.word(0, s, h == 1));
          if constexpr (planes == 2)
            acc[1][b] =
                tl_gemm_vrmpy<false>(acc[1][b], u[h], row.word(1, s, h == 1));
        }
      }
    }
    // The group's int32 dots, weighed by its scales into the fp32 sums: a
    // chain of products added to one accumulator each. A's factor joins B's
    // here only when A has one per group (GroupA); else at the end.
    HVX_Vector scale[NB];
    tl_gemm_half_rows_sf<NB>(scale, sfb + ((g0 + g) / sfb_per) * ld_sfb);
    if constexpr (GroupA) {
      float a_scale = sfa[(g0 + g) / sfa_per];
      int32_t a_bits;
      __builtin_memcpy(&a_bits, &a_scale, 4);
      HVX_Vector va = Q6_V_vsplat_R(a_bits);
#pragma clang loop unroll(full)
      for (int b = 0; b < NB; ++b)
        scale[b] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(scale[b], va));
    }
    HVX_Vector offset;
    if constexpr (BSigned)
      offset = Q6_V_vsplat_R((BBits == 4 ? 8 : 128) * sum[g]);
#pragma clang loop unroll(full)
    for (int b = 0; b < NB; ++b) {
      HVX_Vector r = acc[0][b];
      if constexpr (planes == 2)
        r = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(r, 8), acc[1][b]);
      if constexpr (BSigned)
        r = Q6_Vw_vsub_VwVw(r, offset);
      HVX_Vector p = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vw(r), scale[b]);
      total[b] = Q6_Vqf32_vadd_Vqf32Vqf32(total[b], p);
    }
  }
}

// An unsigned B needs no group sums.
struct tl_gemm_int_no_sums {
  int32_t sum[1] = {0};
};

// The panels an accumulator block of tl_hvx_gemm_int_scaled holds: 4, or the
// largest of 3, 2 and 1 that divides the panels.
constexpr int tl_gemm_int_block(int panels) {
  return panels % 4 == 0 ? 4 : panels % 3 == 0 ? 3 : panels % 2 == 0 ? 2 : 1;
}

template <int M, int N, int K, int BBits, bool BSigned, int LdA, int LdC,
          int GA, int GB, bool AlignC, typename TA>
TL_GEMM void tl_hvx_gemm_int_scaled(float *C, const TA *A, const void *B_,
                                    int panel_bytes, const float *sfa,
                                    int ld_sfa, const half *sfb, int ld_sfb,
                                    bool clear) {
  const uint8_t *B = static_cast<const uint8_t *>(B_);
  static_assert(std::is_same_v<TA, int8_t> || std::is_same_v<TA, uint8_t> ||
                    std::is_same_v<TA, int16_t>,
                "HVX block-scaled GEMM: A is int8, uint8 or int16");
  static_assert(!BSigned || std::is_same_v<TA, int16_t>,
                "HVX block-scaled GEMM: a signed B takes an int16 A");
  constexpr int lane = 32 / BBits;
  constexpr int Group = GA < GB ? GA : GB;
  static_assert(N % 32 == 0 && K % GA == 0 && K % GB == 0 &&
                    Group % lane == 0 && GA % Group == 0 && GB % Group == 0,
                "HVX block-scaled GEMM: whole panels and whole scale groups");
  // Columns per staged chunk of A's row, whole groups; panels per block.
  constexpr int KC = Group >= 512 ? Group : tl_gemm_min(K, 512 / Group * Group);
  constexpr int NB = tl_gemm_int_block(N / 32);
  constexpr bool RowsC = AlignC && LdC % 32 == 0;
  tl_gemm_int_row<TA, BBits, KC> row;
#pragma clang loop unroll(disable)
  for (int i = 0; i < M; ++i) {
    const TA *a = A + i * LdA;
    float *c0 = C + i * LdC;
    // A signed B subtracts its offset times each group's sum of the row.
    std::conditional_t<BSigned, tl_gemm_int_group_sums<K, Group>,
                       tl_gemm_int_no_sums>
        sums;
    if constexpr (BSigned)
      sums.compute(a);
    const int32_t *sum = sums.sum;
#pragma clang loop unroll(disable)
    for (int p = 0; p < N / 32; p += NB) {
      HVX_Vector total[NB];
#pragma clang loop unroll(full)
      for (int b = 0; b < NB; ++b)
        total[b] = Q6_V_vzero();
#pragma clang loop unroll(disable)
      for (int k0 = 0; k0 < K; k0 += KC) {
        int k = tl_gemm_min(KC, K - k0);
        row.stage(a + k0, k, false);
        tl_gemm_int_panels_scaled<TA, BBits, BSigned, NB, KC, Group, (GA < K)>(
            total, row, BSigned ? sum + k0 / Group : sum,
            B + k0 / lane * 128 + p * panel_bytes, panel_bytes, k0 / Group,
            k / Group, sfa + i * ld_sfa, GA / Group, sfb + 32 * p, ld_sfb,
            GB / Group);
      }
      // A's one factor for the whole row, when it has one.
      HVX_Vector va = Q6_V_vzero();
      if constexpr (GA >= K) {
        float a_scale = sfa[i * ld_sfa];
        int32_t a_bits;
        __builtin_memcpy(&a_bits, &a_scale, 4);
        va = Q6_V_vsplat_R(a_bits);
      }
#pragma clang loop unroll(full)
      for (int b = 0; b < NB; ++b) {
        HVX_Vector r = tl_gemm_result(total[b]);
        if constexpr (GA >= K)
          r = tl_gemm_result(Q6_Vqf32_vmpy_VsfVsf(r, va));
        if (!clear)
          r = tl_gemm_add_sf(r, tl_hvx_load<RowsC>(c0 + 32 * (p + b)));
        tl_hvx_store<RowsC>(c0 + 32 * (p + b), r);
      }
    }
  }
}
