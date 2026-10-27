#pragma once
// Copies between Crouton matrices in VTCM and strided matrices (T.copy).
//
// A Crouton tile is 32x32 fp16 stored as interleaved row pairs: element (r, c)
// of a tile sits at (r / 2) * 64 + c * 2 + r % 2, so one HVX register holds a
// row pair. A Crouton matrix orders its tiles by rows or by columns
// (src/hexagon/layout/crouton.h). The copies work in matrix coordinates: the
// region [r0, r0 + rows) x [c0, c0 + cols) of the Crouton matrix, and the
// strided matrix element (r0 + i, c0 + j) at m[i * rs + j * cs], which may be
// in DDR or VTCM, fp16 or fp32 (converted on the way).
#include <hexagon_types.h>
#include <stdint.h>
#include <tl_templates/hexagon/common.h>
#include <tl_templates/hexagon/hvx_lanes.h>

#include <type_traits>

#ifndef TL_DEVICE
#define TL_DEVICE static inline __attribute__((always_inline))
#endif

// The offset of matrix element (r, c) in a Crouton matrix with `row_tiles` x
// `col_tiles` tiles.
template <bool ColumnTiles>
TL_DEVICE int tl_crouton_offset(int r, int c, int row_tiles, int col_tiles) {
  int tile = ColumnTiles ? (c >> 5) * row_tiles + (r >> 5)
                         : (r >> 5) * col_tiles + (c >> 5);
  return (tile << 10) + ((r & 31) >> 1) * 64 + (c & 31) * 2 + (r & 1);
}

// The first `n` (<= 32) elements of two rows, interleaved into one register:
// lanes 2j and 2j + 1 are row0[j] and row1[j], as fp16.
template <typename M>
TL_DEVICE HVX_Vector tl_crouton_interleave(const M *row0, const M *row1,
                                           int n) {
  constexpr int B = static_cast<int>(sizeof(M));
  HVX_Vector lo = tl_hvx_load_bytes(row0, n * B);
  HVX_Vector hi = tl_hvx_load_bytes(row1, n * B);
  if constexpr (std::is_same_v<M, half>) {
    return Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -2));
  } else {
    static_assert(std::is_same_v<M, float>, "fp16 or fp32 matrices");
    HVX_Vector zero = Q6_V_vzero();
    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
        Q6_Vqf32_vadd_VsfVsf(hi, zero), Q6_Vqf32_vadd_VsfVsf(lo, zero)));
  }
}

// The inverse: the first `n` lanes of each row of the row pair in `v` to two
// rows.
template <typename M>
TL_DEVICE void tl_crouton_deinterleave(HVX_Vector v, M *row0, M *row1, int n) {
  constexpr int B = static_cast<int>(sizeof(M));
  if constexpr (std::is_same_v<M, half>) {
    HVX_VectorPair d = Q6_W_vdeal_VVR(v, v, -2);
    tl_hvx_store_bytes(row0, Q6_V_lo_W(d), n * B);
    tl_hvx_store_bytes(row1, Q6_V_hi_W(d), n * B);
  } else {
    static_assert(std::is_same_v<M, float>, "fp16 or fp32 matrices");
    HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(v, Q6_Vh_vsplat_R(0x3c00));
    tl_hvx_store_bytes(row0, Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w)), n * B);
    tl_hvx_store_bytes(row1, Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w)), n * B);
  }
}

// One element at a time, through the Crouton offsets.
template <bool Pack, bool ColumnTiles, typename M>
TL_DEVICE void tl_crouton_copy_elements(half *crouton, int row_tiles,
                                        int col_tiles, int r0, int c0, int i0,
                                        int i1, int cols, M *m, int rs,
                                        int cs) {
  for (int i = i0; i < i1; ++i)
    for (int j = 0; j < cols; ++j) {
      half &x = crouton[tl_crouton_offset<ColumnTiles>(r0 + i, c0 + j,
                                                       row_tiles, col_tiles)];
      M &y = m[i * rs + j * cs];
      if constexpr (Pack)
        x = static_cast<half>(y);
      else
        y = static_cast<M>(x);
    }
}

// Row pairs [i0, i1) (even r0 + i0) over whole tiles, columns [0, cols) with
// c0 on a tile boundary and cols a multiple of the step: two fp16 tiles per
// shuffle, or one tile of fp32 rows per conversion. Counted loops, so that
// clang unrolls and pipelines them.
template <bool Pack, bool ColumnTiles, bool Aligned, typename M>
TL_DEVICE void tl_crouton_copy_tiles(half *crouton, int row_tiles,
                                     int col_tiles, int r0, int c0, int i0,
                                     int i1, int cols, M *m, int rs) {
  constexpr bool fp16 = std::is_same_v<M, half>;
  constexpr int step = fp16 ? 64 : 32;
  for (int i = i0; i < i1; i += 2) {
    int r = r0 + i;
    M *row0 = m + i * rs;
    M *row1 = row0 + rs;
    for (int j = 0; j < cols; j += step) {
      half *pair = crouton + tl_crouton_offset<ColumnTiles>(
                                 r, c0 + j, row_tiles, col_tiles);
      if constexpr (fp16) {
        half *next = crouton + tl_crouton_offset<ColumnTiles>(
                                   r, c0 + j + 32, row_tiles, col_tiles);
        if constexpr (Pack) {
          HVX_VectorPair s =
              Q6_W_vshuff_VVR(tl_hvx_load<Aligned>(row1 + j),
                              tl_hvx_load<Aligned>(row0 + j), -2);
          *reinterpret_cast<HVX_Vector *>(pair) = Q6_V_lo_W(s);
          *reinterpret_cast<HVX_Vector *>(next) = Q6_V_hi_W(s);
        } else {
          HVX_VectorPair d =
              Q6_W_vdeal_VVR(*reinterpret_cast<const HVX_Vector *>(next),
                             *reinterpret_cast<const HVX_Vector *>(pair), -2);
          tl_hvx_store<Aligned>(row0 + j, Q6_V_lo_W(d));
          tl_hvx_store<Aligned>(row1 + j, Q6_V_hi_W(d));
        }
      } else if constexpr (Pack) {
        HVX_Vector zero = Q6_V_vzero();
        HVX_Vector q0 =
            Q6_Vqf32_vadd_VsfVsf(tl_hvx_load<Aligned>(row0 + j), zero);
        HVX_Vector q1 =
            Q6_Vqf32_vadd_VsfVsf(tl_hvx_load<Aligned>(row1 + j), zero);
        *reinterpret_cast<HVX_Vector *>(pair) =
            Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1, q0));
      } else {
        HVX_VectorPair w =
            Q6_Wqf32_vmpy_VhfVhf(*reinterpret_cast<const HVX_Vector *>(pair),
                                 Q6_Vh_vsplat_R(0x3c00));
        tl_hvx_store<Aligned>(row0 + j, Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w)));
        tl_hvx_store<Aligned>(row1 + j, Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w)));
      }
    }
  }
}

// Matrix rows contiguous (cs == 1): one register per row pair and tile.
template <bool Pack, bool ColumnTiles, typename M>
TL_DEVICE void tl_crouton_copy_rows(half *crouton, int row_tiles, int col_tiles,
                                    int r0, int c0, int rows, int cols, M *m,
                                    int rs) {
  // Rows outside whole row pairs: a leading odd row and a trailing one.
  int first = r0 & 1;
  int last = first + ((rows - first) & ~1);
  tl_crouton_copy_elements<Pack, ColumnTiles>(crouton, row_tiles, col_tiles, r0,
                                              c0, 0, first, cols, m, rs, 1);
  tl_crouton_copy_elements<Pack, ColumnTiles>(crouton, row_tiles, col_tiles, r0,
                                              c0, last, rows, cols, m, rs, 1);
  constexpr int step = std::is_same_v<M, half> ? 64 : 32;
  if ((c0 & 31) == 0 && cols % step == 0) {
    if ((reinterpret_cast<uintptr_t>(m) & 127) == 0 &&
        (rs * static_cast<int>(sizeof(M))) % 128 == 0)
      tl_crouton_copy_tiles<Pack, ColumnTiles, true>(
          crouton, row_tiles, col_tiles, r0, c0, first, last, cols, m, rs);
    else
      tl_crouton_copy_tiles<Pack, ColumnTiles, false>(
          crouton, row_tiles, col_tiles, r0, c0, first, last, cols, m, rs);
    return;
  }
  for (int i = first; i < last; i += 2) {
    int r = r0 + i;
    M *row0 = m + i * rs;
    M *row1 = row0 + rs;
    for (int j = 0; j < cols;) {
      int c = c0 + j;
      int lane = c & 31;
      int n = 32 - lane < cols - j ? 32 - lane : cols - j;
      // The row pair's register, at lane 0 of its tile.
      half *pair = crouton + tl_crouton_offset<ColumnTiles>(
                                 r, c - lane, row_tiles, col_tiles);
      if constexpr (std::is_same_v<M, half>) {
        if (lane == 0 && cols - j >= 64) {
          // fp16 rows of 64 columns fill two tiles' registers in one shuffle.
          half *next = crouton + tl_crouton_offset<ColumnTiles>(
                                     r, c + 32, row_tiles, col_tiles);
          if constexpr (Pack) {
            HVX_VectorPair s = Q6_W_vshuff_VVR(tl_hvx_loadu(row1 + j),
                                               tl_hvx_loadu(row0 + j), -2);
            *reinterpret_cast<HVX_Vector *>(pair) = Q6_V_lo_W(s);
            *reinterpret_cast<HVX_Vector *>(next) = Q6_V_hi_W(s);
          } else {
            HVX_VectorPair d =
                Q6_W_vdeal_VVR(*reinterpret_cast<const HVX_Vector *>(next),
                               *reinterpret_cast<const HVX_Vector *>(pair), -2);
            tl_hvx_storeu(row0 + j, Q6_V_lo_W(d));
            tl_hvx_storeu(row1 + j, Q6_V_hi_W(d));
          }
          j += 64;
          continue;
        }
      }
      if constexpr (Pack) {
        tl_hvx_store_bytes(pair + 2 * lane,
                           tl_crouton_interleave(row0 + j, row1 + j, n), 4 * n);
      } else {
        HVX_Vector v = *reinterpret_cast<const HVX_Vector *>(pair);
        tl_crouton_deinterleave(Q6_V_vror_VR(v, 4 * lane), row0 + j, row1 + j,
                                n);
      }
      j += n;
    }
  }
}

// One butterfly stage of a 32x32 transpose of 32-bit words.
template <int Step>
TL_DEVICE void tl_crouton_transpose_stage(HVX_Vector (&v)[32]) {
#pragma clang loop unroll(full)
  for (int base = 0; base < 32; base += 2 * Step) {
#pragma clang loop unroll(full)
    for (int j = 0; j < Step; ++j) {
      HVX_VectorPair p =
          Q6_W_vshuff_VVR(v[base + j + Step], v[base + j], -4 * Step);
      v[base + j] = Q6_V_lo_W(p);
      v[base + j + Step] = Q6_V_hi_W(p);
    }
  }
}

// Packs 64 matrix rows x 32 columns of a column-contiguous fp16 matrix
// (rs == 1): the 32 columns, 64 halves each, are 32 registers of row-pair
// words, whose word transpose is the 32 row-pair registers of two tiles.
// `second` is the offset from the first tile to the second, 32 rows down.
template <bool Aligned>
TL_DEVICE void tl_crouton_pack_transposed_block(half *tile, int second,
                                                const half *m, int cs) {
  HVX_Vector v[32];
#pragma clang loop unroll(full)
  for (int c = 0; c < 32; ++c)
    v[c] = tl_hvx_load<Aligned>(m + c * cs);
  tl_crouton_transpose_stage<1>(v);
  tl_crouton_transpose_stage<2>(v);
  tl_crouton_transpose_stage<4>(v);
  tl_crouton_transpose_stage<8>(v);
  tl_crouton_transpose_stage<16>(v);
#pragma clang loop unroll(full)
  for (int p = 0; p < 32; ++p) {
    // The butterfly leaves row pair p in register bitreverse5(p).
    int at = ((p & 1) << 4) | ((p & 2) << 2) | (p & 4) | ((p & 8) >> 2) |
             ((p & 16) >> 4);
    *reinterpret_cast<HVX_Vector *>(tile + (p >> 4) * second + (p & 15) * 64) =
        v[at];
  }
}

// Packs the blocks of 64 rows x 32 columns in [0, rows) x [0, cols) of a
// column-contiguous fp16 matrix, r0 and c0 on tile boundaries. One block at a
// time: its 32 registers fill the register file.
template <bool ColumnTiles, bool Aligned>
TL_DEVICE void
tl_crouton_pack_transposed(half *crouton, int row_tiles, int col_tiles, int r0,
                           int c0, int rows, int cols, const half *m, int cs) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < rows; i += 64)
#pragma clang loop unroll(disable)
    for (int j = 0; j < cols; j += 32) {
      int at =
          tl_crouton_offset<ColumnTiles>(r0 + i, c0 + j, row_tiles, col_tiles);
      int second = tl_crouton_offset<ColumnTiles>(r0 + i + 32, c0 + j,
                                                  row_tiles, col_tiles) -
                   at;
      tl_crouton_pack_transposed_block<Aligned>(crouton + at, second,
                                                m + i + j * cs, cs);
    }
}

// Copies the region [r0, r0 + rows) x [c0, c0 + cols) between a Crouton
// matrix and the strided matrix m (see the top of this file).
template <bool Pack, bool ColumnTiles, typename M>
TL_DEVICE void tl_crouton_copy(half *crouton, int row_tiles, int col_tiles,
                               int r0, int c0, int rows, int cols, M *m, int rs,
                               int cs) {
  if (cs == 1) {
    tl_crouton_copy_rows<Pack, ColumnTiles>(crouton, row_tiles, col_tiles, r0,
                                            c0, rows, cols, m, rs);
    return;
  }
  if constexpr (Pack && std::is_same_v<M, half>) {
    if (rs == 1 && (r0 & 31) == 0 && (c0 & 31) == 0) {
      // Whole blocks of 64 rows x 32 columns transpose; the rest is
      // copied element by element.
      int block_rows = rows & ~63, block_cols = cols & ~31;
      if ((reinterpret_cast<uintptr_t>(m) & 127) == 0 &&
          (cs * static_cast<int>(sizeof(half))) % 128 == 0)
        tl_crouton_pack_transposed<ColumnTiles, true>(
            crouton, row_tiles, col_tiles, r0, c0, block_rows, block_cols, m,
            cs);
      else
        tl_crouton_pack_transposed<ColumnTiles, false>(
            crouton, row_tiles, col_tiles, r0, c0, block_rows, block_cols, m,
            cs);
      tl_crouton_copy_elements<Pack, ColumnTiles>(crouton, row_tiles, col_tiles,
                                                  r0, c0, block_rows, rows,
                                                  cols, m, 1, cs);
      tl_crouton_copy_elements<Pack, ColumnTiles>(
          crouton, row_tiles, col_tiles, r0, c0 + block_cols, 0, block_rows,
          cols - block_cols, m + block_cols * cs, 1, cs);
      return;
    }
  }
  tl_crouton_copy_elements<Pack, ColumnTiles>(crouton, row_tiles, col_tiles, r0,
                                              c0, 0, rows, cols, m, rs, cs);
}
