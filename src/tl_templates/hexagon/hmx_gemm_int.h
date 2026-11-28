#pragma once

#include <tl_templates/hexagon/hmx_gemm.h>

// Register j in each byte plane holds four rows, interleaved by column.
// Gather even and odd rows while joining bytes, then join halfwords to
// leave one row of 32 accumulator words in each register.
TL_HMX_INLINE void tl_hmx_int_rows(const HVX_Vector *planes,
                                   HVX_Vector (&rows)[4]) {
  HVX_Vector p0 = planes[0], p1 = planes[16];
  HVX_Vector p2 = planes[32], p3 = planes[48];
  HVX_Vector even_lo = Q6_Vb_vshuffe_VbVb(p1, p0);
  HVX_Vector odd_lo = Q6_Vb_vshuffo_VbVb(p1, p0);
  HVX_Vector even_hi = Q6_Vb_vshuffe_VbVb(p3, p2);
  HVX_Vector odd_hi = Q6_Vb_vshuffo_VbVb(p3, p2);
  rows[0] = Q6_Vh_vshuffe_VhVh(even_hi, even_lo);
  rows[1] = Q6_Vh_vshuffe_VhVh(odd_hi, odd_lo);
  rows[2] = Q6_Vh_vshuffo_VhVh(even_hi, even_lo);
  rows[3] = Q6_Vh_vshuffo_VhVh(odd_hi, odd_lo);
}

TL_HMX_INLINE void tl_hmx_int_store(int32_t *c, int ldc,
                                    const HVX_Vector *planes, bool clear) {
  for (int j = 0; j < 16; ++j) {
    HVX_Vector rows[4];
    tl_hmx_int_rows(planes + j, rows);
#pragma clang loop unroll(full)
    for (int r = 0; r < 4; ++r) {
      int32_t *dst = c + (4 * j + r) * ldc;
      HVX_Vector v = rows[r];
      if (!clear)
        v = Q6_Vw_vadd_VwVw(v, tl_hvx_loadu(dst));
      tl_hvx_storeu(dst, v);
    }
  }
}

TL_HMX_INLINE void tl_hmx_int_bias(const HVX_Vector *config, int plane,
                                   int bank) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(config + 2 * plane) | bank;
  asm volatile("bias = mxmem2(%0)" ::"r"(addr) : "memory");
}

template <bool Paired>
TL_HMX_INLINE void tl_hmx_int_mac(int K, const uint8_t *a, const int8_t *b) {
  constexpr int weight_tile_bytes = Paired ? 2048 : 1024;
  asm volatile("mxclracc" ::: "memory");
  for (int k = 0; k < K / 32; k += 32) {
    int n = K / 32 - k < 32 ? K / 32 - k : 32;
    const uint8_t *ap = a + k * 2048;
    const int8_t *bp = b + k * weight_tile_bytes;
    unsigned ar = n * 2048u - 1;
    unsigned wr = ((n * weight_tile_bytes - 128u) & ~127u) | 0x7f;
    if constexpr (Paired)
      asm volatile("{ activation.ub = mxmem(%0, %1):deep\n"
                   "  weight.b = mxmem(%2, %3):deep }" ::"r"(ap),
                   "r"(ar), "r"(bp), "r"(wr)
                   : "memory");
    else
      asm volatile("{ activation.ub = mxmem(%0, %1):deep\n"
                   "  weight.b = mxmem(%2, %3) }" ::"r"(ap),
                   "r"(ar), "r"(bp), "r"(wr)
                   : "memory");
  }
}

TL_HMX_INLINE void tl_hmx_int_convert(HVX_Vector *out, int bank, bool retain) {
  int control = (1 << 15) | (bank << 12) | 2 | retain;
  asm volatile("cvt.ub = acc(%0)\n mxmem(%1, %2) = cvt" ::"r"(control),
               "r"(out), "r"(0xf00)
               : "memory");
}

TL_HMX_INLINE void tl_hmx_int_exact_config(HVX_Vector *banks) {
  for (int i = 0; i < 4; ++i) {
    banks[2 * i] = Q6_V_vsplat_R((24 - 8 * i) << 10);
    banks[2 * i + 1] = Q6_V_vzero();
  }
}

template <bool Shared, bool Paired = false>
TL_HMX_GEMM void
tl_hexagon_hmx_gemm_int(int M, int N, int K, int32_t *c, int ldc,
                        const uint8_t *a, int a_row_tile_bytes, const int8_t *b,
                        int panel_bytes, bool clear, uint8_t *stage,
                        uint32_t *config, bool reload_banks = false) {
  auto *banks = reinterpret_cast<HVX_Vector *>(config);
  tl_hmx_int_exact_config(banks);
  int32_t *previous = nullptr;
  auto acquire = [&]() {
    tl_hmx_unit_acquire();
#if __HVX_ARCH__ >= 79
    // Banks belong to the unit. Another worker may run a different conversion
    // between this worker's tiles and replace the scales and biases.
    if (!previous || reload_banks)
      for (int i = 0; i < 4; ++i)
        tl_hmx_int_bias(banks, i, i);
#endif
  };
  if constexpr (!Shared)
    acquire();

  auto *planes = reinterpret_cast<HVX_Vector *>(stage);
  constexpr int groups = Paired ? 2 : 1;
  constexpr int width = groups * 32;
  int slot = 0;
  auto store_previous = [&]() {
    for (int group = 0; group < groups; ++group)
      tl_hmx_int_store(previous + group * 32, ldc,
                       planes + ((slot ^ 1) * groups + group) * 64, clear);
  };
  for (int mt = 0; mt < M / 64; ++mt) {
    for (int nt = 0; nt < N / width; ++nt, slot ^= 1) {
      if constexpr (Shared)
        acquire();
      tl_hmx_int_mac<Paired>(K, a + mt * a_row_tile_bytes,
                             b + nt * panel_bytes);
      if constexpr (!Shared)
        if (previous)
          store_previous();
      // Releasing the fourth byte clears this accumulator and selects the next.
      for (int group = 0; group < groups; ++group) {
        for (int i = 0; i < 4; ++i) {
#if __HVX_ARCH__ >= 79
          int bank = i;
#else
          int bank = 0;
          tl_hmx_int_bias(banks, i, bank);
#endif
          auto *out = planes + (slot * groups + group) * 64 + i * 16;
          tl_hmx_int_convert(out, bank, i < 3);
        }
      }
      if constexpr (Shared) {
        tl_hmx_unit_release();
        if (previous)
          store_previous();
      }
      previous = c + mt * 64 * ldc + nt * width;
    }
  }
  if constexpr (!Shared)
    tl_hmx_unit_release();
  store_previous();
}

// Two planes per group. vdeal joins the two column groups into full rows.
// With a single group, only the first 32 halfwords of each row are stored.
template <bool Paired>
TL_HMX_INLINE void tl_hmx_requant_store(int16_t *c, int ldc,
                                        const HVX_Vector *planes) {
  for (int q = 0; q < 16; ++q) {
    const HVX_Vector *l = planes + q, *r = Paired ? l + 32 : l;
    if constexpr (!Paired) {
      if (ldc == 32) {
        // Four half-register rows fill two registers when the rows abut.
        HVX_VectorPair pairs =
            Q6_W_vdeal_VVR(Q6_Vb_vshuffo_VbVb(l[16], l[0]),
                           Q6_Vb_vshuffe_VbVb(l[16], l[0]), -2);
        tl_hvx_storeu(c + 128 * q, Q6_V_lo_W(pairs));
        tl_hvx_storeu(c + 128 * q + 64, Q6_V_hi_W(pairs));
        continue;
      }
    }
    HVX_VectorPair even = Q6_W_vdeal_VVR(Q6_Vb_vshuffe_VbVb(r[16], r[0]),
                                         Q6_Vb_vshuffe_VbVb(l[16], l[0]), -2);
    HVX_VectorPair odd = Q6_W_vdeal_VVR(Q6_Vb_vshuffo_VbVb(r[16], r[0]),
                                        Q6_Vb_vshuffo_VbVb(l[16], l[0]), -2);
    HVX_Vector rows[] = {Q6_V_lo_W(even), Q6_V_lo_W(odd), Q6_V_hi_W(even),
                         Q6_V_hi_W(odd)};
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
      auto *dst = c + (4 * q + i) * ldc;
      if constexpr (Paired)
        tl_hvx_storeu(dst, rows[i]);
      else
        tl_hvx_store_bytes(dst, rows[i], 64);
    }
  }
}

TL_HMX_INLINE void tl_hmx_requant_exact_store(int16_t *c, int ldc,
                                              const HVX_Vector *planes,
                                              HVX_Vector bias,
                                              HVX_Vector shift) {
  for (int q = 0; q < 16; ++q) {
    HVX_Vector rows[4];
    tl_hmx_int_rows(planes + q, rows);
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
      HVX_Vector x = Q6_Vw_vasr_VwVw(Q6_Vw_vadd_VwVw(rows[i], bias), shift);
      tl_hvx_store_bytes(c + (4 * q + i) * ldc, Q6_Vh_vpacke_VwVw(x, x), 64);
    }
  }
}

// Prepare both bytes of each column from the same bias. Out-of-range shifts
// use the four exact word bytes and the logical map in HVX registers.
template <int Groups>
TL_HMX_INLINE bool
tl_hmx_requant_config(const int32_t *bias, const int32_t *shift,
                      HVX_Vector (&biases)[Groups],
                      HVX_Vector (&shifts)[Groups], HVX_Vector *banks) {
  for (int g = 0; g < Groups; ++g) {
    biases[g] = tl_hvx_loadu(bias + 32 * g);
    shifts[g] = tl_hvx_loadu(shift + 32 * g);
  }
  HVX_Vector limit = Q6_V_vsplat_R(16);
  HVX_VectorPred outside = Q6_Q_vcmp_gt_VuwVuw(shifts[0], limit);
  if constexpr (Groups == 2)
    outside = Q6_Q_or_QQ(outside, Q6_Q_vcmp_gt_VuwVuw(shifts[1], limit));
  bool exact = Q6_R_vextract_VR(Q6_Vw_prefixsum_Q(outside), 124) != 0;
  if (exact) {
    tl_hmx_int_exact_config(banks);
  } else {
    for (int i = 0; i < 2 * Groups; ++i) {
      HVX_Vector exponent =
          Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(24 - 8 * (i % 2)), shifts[i / 2]);
      banks[2 * i] = Q6_Vw_vasl_VwR(exponent, 10);
      banks[2 * i + 1] = biases[i / 2];
    }
  }
  return exact;
}

template <bool Shared, bool Paired>
TL_HMX_GEMM void tl_hexagon_hmx_gemm_int_requant(
    int M, int N, int K, int16_t *c, int ldc, const uint8_t *a,
    int a_row_tile_bytes, const int8_t *b, int panel_bytes, const int32_t *bias,
    const int32_t *shift, uint8_t *stage, uint32_t *config) {
  constexpr int groups = Paired ? 2 : 1, width = groups * 32;
  auto *banks = reinterpret_cast<HVX_Vector *>(config);
  auto *planes = reinterpret_cast<HVX_Vector *>(stage);
  for (int nt = 0; nt < N / width; ++nt) {
    HVX_Vector biases[groups], shifts[groups];
    bool exact = tl_hmx_requant_config(bias + nt * width, shift + nt * width,
                                       biases, shifts, banks);
    auto acquire = [&] {
      tl_hmx_unit_acquire();
#if __HVX_ARCH__ >= 79
      // All banks belong to the unit, including between workers of this GEMM.
      for (int i = 0; i < (exact ? 4 : 2 * groups); ++i)
        tl_hmx_int_bias(banks, i, i);
#endif
    };
    // Four planes of scratch also hold two ordinary two-byte results.
    // An exclusive caller joins the previous tile during the next MAC.
    auto fast_planes = [&](int mt) {
      return planes + (Shared ? 0 : mt % 2) * groups * 32;
    };
    auto store_fast = [&](int mt) {
      tl_hmx_requant_store<Paired>(c + mt * 64 * ldc + nt * width, ldc,
                                   fast_planes(mt));
    };
    if constexpr (!Shared)
      acquire();
    for (int mt = 0; mt < M / 64; ++mt) {
      if constexpr (Shared)
        acquire();
      tl_hmx_int_mac<Paired>(K, a + mt * a_row_tile_bytes,
                             b + nt * panel_bytes);
      if constexpr (!Shared)
        if (!exact && mt)
          store_fast(mt - 1);
      HVX_Vector *tile = exact ? planes : fast_planes(mt);
      int bytes = exact ? 4 : 2;
      for (int g = 0; g < groups; ++g)
        for (int i = 0; i < bytes; ++i) {
          int index = exact ? i : 2 * g + i;
#if __HVX_ARCH__ >= 79
          int bank = index;
#else
          int bank = 0;
          tl_hmx_int_bias(banks, index, bank);
#endif
          tl_hmx_int_convert(tile + (bytes * g + i) * 16, bank, i + 1 < bytes);
        }
      if constexpr (Shared)
        tl_hmx_unit_release();
      if (exact) {
        int16_t *dst = c + mt * 64 * ldc + nt * width;
        for (int g = 0; g < groups; ++g)
          tl_hmx_requant_exact_store(dst + 32 * g, ldc, planes + 64 * g,
                                     biases[g], shifts[g]);
      } else if constexpr (Shared) {
        store_fast(mt);
      }
    }
    if constexpr (!Shared) {
      tl_hmx_unit_release();
      if (!exact)
        store_fast(M / 64 - 1);
    }
  }
}
