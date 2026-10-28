#pragma once
// GEMMs on HMX that are functions of their own: T.gemm_blockscaled of an fp16
// A. Like the HVX GEMMs (TL_GEMM, hvx_gemm.h), the calls of one configuration
// share one copy of the code, so a megakernel that multiplies in hundreds of
// places, a prefill's every block, stays small enough to compile and to run
// from the instruction cache.
//
// The `tl_hexagon_hmx` prefix makes runtime/fastrpc.py set up the HMX session.
#include <tl_templates/hexagon/crouton.h>
#include <tl_templates/hexagon/hmx.h>

#define TL_HMX_GEMM static __attribute__((noinline))

// The conversion config HMX reads, 32 scale words then 32 bias words: each
// scale word carries fp16 1.0 in its low half, every bias is 0. One register
// each: scalar stores into VTCM are slow.
TL_HMX_INLINE void tl_hmx_unit_config(uint32_t *config) {
  *reinterpret_cast<HVX_Vector *>(config) = Q6_V_vsplat_R(0x3C00);
  *reinterpret_cast<HVX_Vector *>(config + 32) = Q6_V_vzero();
}

// C (+)= (A * sfa) (B * sfb)^T on HMX: an fp16 A [M, K] in the HMX activation
// layout, row tile mt at a + mt * a_row_tile; a 4-bit B [N, K] in the panel
// layout, 32 rows a panel_bytes apart; sfb its fp16 factors, row j's of group
// g of gb columns at sfb[g * ld_sfb + j]; and sfa an fp32 factor per row of A,
// row i's at sfa[i * ld_sfa] (Scaled). C is row-major (Rows), element (i, j)
// at c[i * ldc + j], fp32 or fp16, or fp16 Crouton tiles, tile (mt, nt) at
// c + (mt * ldc + nt) * 1024; Aligned: its rows start on register (fp32) or
// half-register (fp16) boundaries. clear overwrites C.
//
// HVX dequantizes B 32 rows at a time, a column of output tiles, into a column
// of fp16 weight tiles in w (two of K / 32 tiles each, alternating), and HMX
// multiplies every row tile of A by it. One program dequantizes a slice of the
// next column and runs the previous tile's epilogue while HMX multiplies each
// tile. Shared: the workers of a pool take turns at HMX, a tile each, and each
// runs its epilogue and slice outside its turn. stage holds two staging tiles,
// config the conversion config.
template <bool Signed, bool Rows, bool Aligned, bool Scaled, bool Shared,
          typename D>
TL_HMX_GEMM void tl_hexagon_hmx_gemm_blockscaled(
    int M, int N, int K, D *c, int ldc, const half *a, int a_row_tile,
    const void *b, int panel_bytes, const half *sfb, int ld_sfb, int gb,
    const float *sfa, int ld_sfa, bool clear, half *w, half *stage,
    uint32_t *config) {
  const int MT = M / TL_HMX_T, NT = N / TL_HMX_T, KT = K / TL_HMX_T;
  const int column = K * TL_HMX_T; // halves of a column of weight tiles
  // A column is K / 8 panel registers, dequantized in MT slices.
  const int registers = K / 8;
  const int per_slice = (registers + MT - 1) / MT;
  const uint8_t *panels = static_cast<const uint8_t *>(b);

  // The tiles go by columns, (mt, nt) for nt, for mt, with no division by
  // the run-time counts: Hexagon divides in software.
  auto dequant = [&](int nt, int v0, int n) {
    tl_crouton_dequant4<Signed>(w + (nt & 1) * column,
                                panels + nt * panel_bytes, sfb + nt * TL_HMX_T,
                                ld_sfb, gb, v0, n);
  };
  // Slice mt of column nt + 1, alongside tile (mt, nt).
  auto slice = [&](int mt, int nt) {
    int v0 = mt * per_slice;
    if (nt + 1 < NT && v0 < registers)
      dequant(nt + 1, v0,
              registers - v0 < per_slice ? registers - v0 : per_slice);
  };
  auto multiply = [&](int mt, int nt) {
    const half *row = a + mt * a_row_tile;
    const half *col = w + (nt & 1) * column;
    for (int k = 0; k < KT; k += 32)
      tl_hmx_mac_tiles(row + k * TL_HMX_TILE_ELMS, col + k * TL_HMX_TILE_ELMS,
                       KT - k < 32 ? KT - k : 32);
  };
  // Tile (mt, nt) from staging tile `slot` into C.
  auto epilogue = [&](int mt, int nt, int slot) {
    const half *tile = stage + slot * TL_HMX_TILE_ELMS;
    D *dst = Rows ? c + mt * TL_HMX_T * ldc + nt * TL_HMX_T
                  : c + (mt * ldc + nt) * TL_HMX_TILE_ELMS;
    const float *scale = Scaled ? sfa + mt * TL_HMX_T * ld_sfa : nullptr;
    if (clear)
      tl_crouton_tile_store<false, Rows, Aligned, Scaled>(dst, ldc, tile, scale,
                                                          ld_sfa);
    else
      tl_crouton_tile_store<true, Rows, Aligned, Scaled>(dst, ldc, tile, scale,
                                                         ld_sfa);
  };

  tl_hmx_unit_config(config);
  dequant(0, 0, registers);
  int slot = 0;
  if constexpr (Shared) {
    for (int nt = 0; nt < NT; ++nt)
      for (int mt = 0; mt < MT; ++mt, slot ^= 1) {
        tl_hmx_unit_acquire();
        tl_hmx_set_scales(config);
        tl_hmx_clear_acc();
        multiply(mt, nt);
        tl_hmx_convert_acc(2);
        tl_hmx_store_cvt(stage + slot * TL_HMX_TILE_ELMS);
        tl_hmx_unit_release();
        epilogue(mt, nt, slot);
        slice(mt, nt);
      }
  } else {
    tl_hmx_unit_acquire();
    tl_hmx_set_scales(config);
    // The previous tile, whose epilogue runs while HMX multiplies.
    int last_mt = -1, last_nt = 0;
    for (int nt = 0; nt < NT; ++nt)
      for (int mt = 0; mt < MT; ++mt, slot ^= 1) {
        tl_hmx_clear_acc();
        multiply(mt, nt);
        // HVX dequantizes and runs the previous tile's epilogue while HMX
        // multiplies.
        slice(mt, nt);
        if (last_mt >= 0)
          epilogue(last_mt, last_nt, slot ^ 1);
        tl_hmx_convert_acc(2);
        tl_hmx_store_cvt(stage + slot * TL_HMX_TILE_ELMS);
        last_mt = mt;
        last_nt = nt;
      }
    tl_hmx_unit_release();
    epilogue(last_mt, last_nt, slot ^ 1);
  }
}
