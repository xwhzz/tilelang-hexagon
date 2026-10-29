#pragma once
// The GEMMs on HMX: T.gemm of fp16 operands in VTCM, and T.gemm_blockscaled of
// an fp16 A by a 4-bit B, one tile loop (tl_hmx_gemm) over two sources of
// weight tiles.
//
// T.gemm inlines into its call site, where its shapes and strides are
// constants: a kernel calls it in a few places, and out of line a small call
// costs its loop's arithmetic (on v79, a 128 x 128 x 64 call took 4 to 6% more
// cycles).
// T.gemm_blockscaled is a function of its own: like the HVX GEMMs (TL_GEMM,
// hvx_gemm.h), the calls of one configuration share one copy of the code, so
// a megakernel that multiplies in hundreds of places, a prefill's every
// block, stays small enough to compile and to run from the instruction cache.

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

// The B of T.gemm: fp16 weight tiles in VTCM, in the HMX weight layout, the K
// tiles of output column nt contiguous from tiles + nt * stride.
struct tl_hmx_weight_tiles {
  const half *tiles;
  int stride;

  const half *column(int nt) const { return tiles + nt * stride; }
  void prepare() const {}
  void slice(int, int) const {}
};

// The B of T.gemm_blockscaled: 4-bit weights [N, K] in the panel layout, 32
// rows a panel_bytes apart, and their fp16 factors, row j's of group g of gb
// columns of K at sfb[g * ld_sfb + j]. HVX dequantizes them 32 rows at a time,
// a column of output tiles, into a column of fp16 weight tiles in w (two of
// K / 32 tiles each, alternating): one column while HMX multiplies the
// other, in slices, one alongside each row tile of A.
template <bool Signed> struct tl_hmx_panel_weights {
  const uint8_t *panels;
  int panel_bytes;
  const half *sfb;
  int ld_sfb, gb;
  half *w;
  int column_halves; // K * 32: one column of weight tiles
  int registers;     // panel registers of a column, K / 8
  int per_slice;     // of them dequantized alongside a row tile
  int columns;       // N / 32

  const half *column(int nt) const { return w + (nt & 1) * column_halves; }
  void dequant(int nt, int v0, int n) const {
    tl_crouton_dequant4<Signed>(w + (nt & 1) * column_halves,
                                panels + nt * panel_bytes, sfb + nt * TL_HMX_T,
                                ld_sfb, gb, v0, n);
  }
  void prepare() const { dequant(0, 0, registers); }
  // Slice mt of column nt + 1, alongside tile (mt, nt).
  void slice(int mt, int nt) const {
    int v0 = mt * per_slice;
    if (nt + 1 < columns && v0 < registers)
      dequant(nt + 1, v0,
              registers - v0 < per_slice ? registers - v0 : per_slice);
  }
};

// C (+)= (A * sfa) B^T on HMX, B a weight source (above): an fp16 A [M, K] in
// the HMX activation layout, row tile mt at a + mt * a_row_tile, its K tiles
// contiguous; sfa an fp32 factor per row of A, row i's at sfa[i * ld_sfa]
// (Scaled). C is row-major (Rows), element (i, j) at c[i * ldc + j], fp32 or
// fp16, or fp16 Crouton tiles, tile (mt, nt) at c + (mt * ldc + nt) * 1024;
// Aligned: its rows start on register (fp32) or half-register (fp16)
// boundaries. clear overwrites C. Direct: an overwritten Crouton C takes HMX's
// tiles as they are; else each tile goes through a staging tile in stage, which
// an HVX epilogue writes or adds into C. config holds the conversion config.
//
// The tiles go by columns, (mt, nt) for nt, for mt, with no division by the
// run-time counts: Hexagon divides in software. One program runs the previous
// tile's epilogue, and B's slice, while HMX multiplies each tile. Shared: the
// workers of a pool take turns at HMX, a tile each, and each runs its epilogue
// and slice outside its turn.
template <bool Rows, bool Aligned, bool Scaled, bool Shared, bool Direct,
          typename D, typename B>
TL_HMX_INLINE void tl_hmx_gemm(int M, int N, int K, D *c, int ldc,
                               const half *a, int a_row_tile, const B &b,
                               const float *sfa, int ld_sfa, bool clear,
                               half *stage, uint32_t *config) {
  static_assert(!Direct || (!Rows && !Scaled),
                "HMX stores only into Crouton tiles, as they are");
  const int MT = M / TL_HMX_T, NT = N / TL_HMX_T, KT = K / TL_HMX_T;

  auto multiply = [&](int mt, int nt) {
    const half *row = a + mt * a_row_tile;
    const half *col = b.column(nt);
    for (int k = 0; k < KT; k += 32)
      tl_hmx_mac_tiles(row + k * TL_HMX_TILE_ELMS, col + k * TL_HMX_TILE_ELMS,
                       KT - k < 32 ? KT - k : 32);
  };
  // Where tile (mt, nt) leaves HMX: C's own tile, or staging tile `slot`.
  auto target = [&](int mt, int nt, int slot) -> half * {
    if constexpr (Direct)
      return reinterpret_cast<half *>(c) + (mt * ldc + nt) * TL_HMX_TILE_ELMS;
    return stage + slot * TL_HMX_TILE_ELMS;
  };
  // Tile (mt, nt) from staging tile `slot` into C.
  auto epilogue = [&](int mt, int nt, int slot) {
    if constexpr (!Direct) {
      const half *tile = stage + slot * TL_HMX_TILE_ELMS;
      D *dst = Rows ? c + mt * TL_HMX_T * ldc + nt * TL_HMX_T
                    : c + (mt * ldc + nt) * TL_HMX_TILE_ELMS;
      const float *scale = Scaled ? sfa + mt * TL_HMX_T * ld_sfa : nullptr;
      if (clear)
        tl_crouton_tile_store<false, Rows, Aligned, Scaled>(dst, ldc, tile,
                                                            scale, ld_sfa);
      else
        tl_crouton_tile_store<true, Rows, Aligned, Scaled>(dst, ldc, tile,
                                                           scale, ld_sfa);
    }
  };

  tl_hmx_unit_config(config);
  b.prepare();
  int slot = 0;
  if constexpr (Shared) {
    // A worker's own MACs run while it holds HMX; when they take longer than
    // an epilogue (K of 32 tiles or more), the previous tile's epilogue runs
    // under them, so that the worker's HVX work outside its turn shrinks.
    const bool under = !Direct && KT >= 32;
    int last_mt = -1, last_nt = 0;
    for (int nt = 0; nt < NT; ++nt)
      for (int mt = 0; mt < MT; ++mt, slot ^= 1) {
        tl_hmx_unit_acquire();
        tl_hmx_set_scales(config);
        tl_hmx_clear_acc();
        multiply(mt, nt);
        if (under && last_mt >= 0)
          epilogue(last_mt, last_nt, slot ^ 1);
        tl_hmx_convert_acc(2);
        tl_hmx_store_cvt(target(mt, nt, slot));
        tl_hmx_unit_release();
        if (!under)
          epilogue(mt, nt, slot);
        b.slice(mt, nt);
        last_mt = mt;
        last_nt = nt;
      }
    if (under)
      epilogue(last_mt, last_nt, slot ^ 1);
  } else {
    tl_hmx_unit_acquire();
    tl_hmx_set_scales(config);
    // The previous tile, whose epilogue runs while HMX multiplies.
    int last_mt = -1, last_nt = 0;
    for (int nt = 0; nt < NT; ++nt)
      for (int mt = 0; mt < MT; ++mt, slot ^= 1) {
        tl_hmx_clear_acc();
        multiply(mt, nt);
        // HVX slices B and runs the previous tile's epilogue while HMX
        // multiplies.
        b.slice(mt, nt);
        if (last_mt >= 0)
          epilogue(last_mt, last_nt, slot ^ 1);
        tl_hmx_convert_acc(2);
        tl_hmx_store_cvt(target(mt, nt, slot));
        last_mt = mt;
        last_nt = nt;
      }
    tl_hmx_unit_release();
    epilogue(last_mt, last_nt, slot ^ 1);
  }
}

// T.gemm on HMX: C (+)= A B^T, B fp16 weight tiles, column nt's at
// b + nt * b_column_tile.
template <bool Rows, bool Aligned, bool Shared, bool Direct, typename D>
TL_HMX_INLINE void
tl_hexagon_hmx_gemm(int M, int N, int K, D *c, int ldc, const half *a,
                    int a_row_tile, const half *b, int b_column_tile,
                    bool clear, half *stage, uint32_t *config) {
  tl_hmx_gemm<Rows, Aligned, false, Shared, Direct>(
      M, N, K, c, ldc, a, a_row_tile, tl_hmx_weight_tiles{b, b_column_tile},
      nullptr, 0, clear, stage, config);
}

// T.gemm_blockscaled on HMX: C (+)= (A * sfa) (B * sfb)^T, B 4-bit weights in
// the panel layout (tl_hmx_panel_weights), dequantized into w.
template <bool Signed, bool Rows, bool Aligned, bool Scaled, bool Shared,
          typename D>
TL_HMX_GEMM void tl_hexagon_hmx_gemm_blockscaled(
    int M, int N, int K, D *c, int ldc, const half *a, int a_row_tile,
    const void *b, int panel_bytes, const half *sfb, int ld_sfb, int gb,
    const float *sfa, int ld_sfa, bool clear, half *w, half *stage,
    uint32_t *config) {
  const int MT = M / TL_HMX_T;
  // A column is K / 8 panel registers, dequantized in MT slices.
  const int registers = K / 8;
  const tl_hmx_panel_weights<Signed> panels{static_cast<const uint8_t *>(b),
                                            panel_bytes,
                                            sfb,
                                            ld_sfb,
                                            gb,
                                            w,
                                            K * TL_HMX_T,
                                            registers,
                                            (registers + MT - 1) / MT,
                                            N / TL_HMX_T};
  tl_hmx_gemm<Rows, Aligned, Scaled, Shared, false>(
      M, N, K, c, ldc, a, a_row_tile, panels, sfa, ld_sfa, clear, stage,
      config);
}
