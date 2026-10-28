#pragma once
// Copies between Crouton matrices in VTCM and strided matrices (T.copy),
// reductions of Crouton matrices along rows or columns (T.reduce_*), the
// stores of HMX output tiles into a gemm's C (T.gemm), and HMX weight tiles
// dequantized from 4-bit panels (T.gemm_blockscaled).
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
#include <tl_templates/hexagon/hvx_reduce.h>

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

// ---------------------------------------------------------------------------
// Reductions (hvx_reduce.h's kinds and transpose tree).
// ---------------------------------------------------------------------------

// Row pair q of tile (r, c), one register.
template <bool ColumnTiles>
TL_DEVICE HVX_Vector tl_crouton_pair(const half *crouton, int row_tiles,
                                     int col_tiles, int r, int c, int q) {
  int tile = ColumnTiles ? c * row_tiles + r : r * col_tiles + c;
  return *reinterpret_cast<const HVX_Vector *>(crouton + tile * 1024 + q * 64);
}

// The two rows of a row pair register as registers of A: fp32 widening splits
// the even lanes (row 2q) from the odd ones (row 2q + 1).
template <int K, typename A>
TL_DEVICE void tl_crouton_rows_of(HVX_Vector pair, tl_hvx_reg<A> *even,
                                  tl_hvx_reg<A> *odd) {
  static_assert(std::is_same_v<A, float>, "fp32 rows");
  HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(pair, Q6_Vh_vsplat_R(0x3c00));
  *even = tl_reduce_operand<K, A>(
      tl_hvx_bitcast<tl_hvx_reg<A>>(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w))));
  *odd = tl_reduce_operand<K, A>(
      tl_hvx_bitcast<tl_hvx_reg<A>>(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w))));
}

// Rows [2 q0, 2 q0 + 2N) of row tile r, every column tile combined, as one
// register of A in the transpose tree's form: rows in runs of lanes, in order.
template <int K, typename A, bool ColumnTiles, int N>
TL_DEVICE tl_hvx_reg<A> tl_crouton_row_tree(const half *crouton, int row_tiles,
                                            int col_tiles, int r, int q0) {
  if constexpr (N == 1) {
    if constexpr (std::is_same_v<A, half>) {
      // Lane 2c + e holds row 2 q0 + e, column c; dealing the halfwords puts
      // the two rows in runs of 32 lanes.
      tl_hvx_reg<A> acc = tl_reduce_identity<K, A>();
      for (int c = 0; c < col_tiles; ++c)
        acc = tl_reduce_combine<K>(
            acc, tl_reduce_operand<K, A>(
                     tl_hvx_bitcast<tl_hvx_reg<A>>(tl_crouton_pair<ColumnTiles>(
                         crouton, row_tiles, col_tiles, r, c, q0))));
      return tl_hvx_bitcast<tl_hvx_reg<A>>(
          Q6_Vh_vdeal_Vh(tl_hvx_bitcast<HVX_Vector>(acc)));
    } else {
      tl_hvx_reg<A> even = tl_reduce_identity<K, A>(), odd = even;
      for (int c = 0; c < col_tiles; ++c) {
        tl_hvx_reg<A> e, o;
        tl_crouton_rows_of<K, A>(tl_crouton_pair<ColumnTiles>(
                                     crouton, row_tiles, col_tiles, r, c, q0),
                                 &e, &o);
        even = tl_reduce_combine<K>(even, e);
        odd = tl_reduce_combine<K>(odd, o);
      }
      return tl_reduce_merge<K, A>(even, odd);
    }
  } else {
    return tl_reduce_merge<K, A>(
        tl_crouton_row_tree<K, A, ColumnTiles, N / 2>(crouton, row_tiles,
                                                      col_tiles, r, q0),
        tl_crouton_row_tree<K, A, ColumnTiles, N / 2>(
            crouton, row_tiles, col_tiles, r, q0 + N / 2));
  }
}

// The columns of column tile c, every row combined: lanes [0, 32) of a
// register of A.
template <int K, typename A, bool ColumnTiles>
TL_DEVICE tl_hvx_reg<A> tl_crouton_columns(const half *crouton, int row_tiles,
                                           int col_tiles, int c) {
  // Two accumulators: even and odd rows, or two chains of row pairs.
  tl_hvx_reg<A> a0 = tl_reduce_identity<K, A>(), a1 = a0;
  for (int r = 0; r < row_tiles; ++r) {
#pragma clang loop unroll_count(2)
    for (int q = 0; q < 16; ++q) {
      HVX_Vector pair =
          tl_crouton_pair<ColumnTiles>(crouton, row_tiles, col_tiles, r, c, q);
      if constexpr (std::is_same_v<A, half>) {
        tl_hvx_reg<A> x =
            tl_reduce_operand<K, A>(tl_hvx_bitcast<tl_hvx_reg<A>>(pair));
        if (q & 1)
          a1 = tl_reduce_combine<K>(a1, x);
        else
          a0 = tl_reduce_combine<K>(a0, x);
      } else {
        tl_hvx_reg<A> e, o;
        tl_crouton_rows_of<K, A>(pair, &e, &o);
        a0 = tl_reduce_combine<K>(a0, e);
        a1 = tl_reduce_combine<K>(a1, o);
      }
    }
  }
  tl_hvx_reg<A> v = tl_reduce_combine<K>(a0, a1);
  if constexpr (std::is_same_v<A, half>) {
    // Lane 2c + e: the two rows of each column, dealt apart and combined.
    HVX_Vector d = Q6_Vh_vdeal_Vh(tl_hvx_bitcast<HVX_Vector>(v));
    v = tl_reduce_combine<K>(
        tl_hvx_bitcast<tl_hvx_reg<A>>(d),
        tl_hvx_bitcast<tl_hvx_reg<A>>(Q6_V_vror_VR(d, 64)));
  }
  return v;
}

// Reduces a Crouton matrix with row_tiles x col_tiles tiles along its rows
// (Rows: one result per row, the columns combined) or its columns, into
// dst[i * dst_stride], accumulating in A. Combines with dst unless `clear`.
template <int K, typename A, bool Rows, bool ColumnTiles, typename D>
TL_DEVICE void tl_crouton_reduce(D *dst, const half *crouton, int row_tiles,
                                 int col_tiles, int dst_stride, bool clear) {
  if constexpr (Rows) {
    for (int r = 0; r < row_tiles; ++r) {
      tl_hvx_reg<A> v = tl_crouton_row_tree<K, A, ColumnTiles, 16>(
          crouton, row_tiles, col_tiles, r, 0);
      // fp16 rows end in runs of two lanes.
      if constexpr (std::is_same_v<A, half>)
        v = tl_reduce_merge<K, A>(v, v);
      tl_reduce_store<K, A>(dst + r * 32 * dst_stride, dst_stride, v, 32,
                            clear);
    }
  } else {
    for (int c = 0; c < col_tiles; ++c)
      tl_reduce_store<K, A>(dst + c * 32 * dst_stride, dst_stride,
                            tl_crouton_columns<K, A, ColumnTiles>(
                                crouton, row_tiles, col_tiles, c),
                            32, clear);
  }
}

// ---------------------------------------------------------------------------
// HMX output tiles (T.gemm): an HMX tile, stored fp16 to a Crouton staging
// tile, written or added into the gemm's C.
// ---------------------------------------------------------------------------

// Writes, or adds (Accumulate), a 64-byte fp16 row at `row` from a register
// that holds the row in both halves. Aligned: the row starts on a
// half-register boundary, so it is the masked half of one aligned block.
template <bool Accumulate, bool Aligned>
TL_DEVICE void tl_crouton_half_row(half *row, HVX_Vector v) {
  if constexpr (Aligned) {
    int off = tl_hvx_block_offset(row);
    HVX_Vector *block = const_cast<HVX_Vector *>(tl_hvx_block_of(row));
    if constexpr (Accumulate)
      v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(*block, v));
    HVX_VectorPred half_at =
        Q6_Q_and_QQn(Q6_Q_vsetq2_R(off + 64), Q6_Q_vsetq_R(off));
    Q6_vmem_QRIV(half_at, block, v);
  } else {
    if constexpr (Accumulate)
      v = Q6_Vhf_equals_Vqf16(
          Q6_Vqf16_vadd_VhfVhf(tl_hvx_load_bytes(row, 64), v));
    tl_hvx_store_bytes(row, v, 64);
  }
}

// The fp16 factors of a tile's 32 rows, scale[i * ld] for row i, one HVX
// conversion for all of them: factor i lands in halfwords 2i and 2i + 1 of
// `h`, so word q of `pairs` below holds row pair q's two.
TL_DEVICE void tl_crouton_row_factors(uint16_t *__restrict__ h,
                                      const float *__restrict__ scale, int ld) {
  HVX_Vector v;
  if (ld == 1) {
    v = tl_hvx_load_bytes(scale, 128);
  } else {
    alignas(128) float f[32];
    for (int i = 0; i < 32; ++i)
      f[i] = scale[i * ld];
    v = *reinterpret_cast<const HVX_Vector *>(f);
  }
  HVX_Vector q = Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vzero());
  *reinterpret_cast<HVX_Vector *>(h) =
      Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q, q));
}

// Row pair q's factors, that of row 2q in the even lanes and that of row
// 2q + 1 in the odd ones, from tl_crouton_row_factors' `h`.
TL_DEVICE HVX_Vector tl_crouton_pair_factors(const uint16_t *h, int q) {
  return Q6_V_vsplat_R(
      static_cast<int>(h[4 * q] | (uint32_t(h[4 * q + 2]) << 16)));
}

// Writes the fp16 Crouton tile `tile` into the 32 x 32 block at `dst`, or adds
// it (Accumulate); Scaled multiplies row i by scale[i * ld], in fp16. Rows: the
// block is row-major, element (i, j) at dst[i * ldc + j], fp16 or fp32;
// Aligned when its rows start on register boundaries (fp32) or half-register
// ones (fp16). Otherwise dst is a Crouton tile, fp16.
template <bool Accumulate, bool Rows, bool Aligned, bool Scaled, typename D>
TL_DEVICE void tl_crouton_tile_store(D *__restrict__ dst, int ldc,
                                     const half *__restrict__ tile,
                                     const float *__restrict__ scale = 0,
                                     int ld = 1) {
  const HVX_Vector *t = reinterpret_cast<const HVX_Vector *>(tile);
  alignas(128) uint16_t h[64];
  if constexpr (Scaled)
    tl_crouton_row_factors(h, scale, ld);
  // Row pair q's factors (1.0 unscaled), and the pair scaled.
  auto factors = [&](int q) {
    if constexpr (Scaled)
      return tl_crouton_pair_factors(h, q);
    else
      return Q6_Vh_vsplat_R(0x3c00);
  };
  auto pair = [&](int q) {
    if constexpr (Scaled)
      return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(t[q], factors(q)));
    else
      return t[q];
  };
  if constexpr (!Rows) {
    static_assert(std::is_same_v<D, half>, "Crouton tiles are fp16");
    HVX_Vector *d = reinterpret_cast<HVX_Vector *>(dst);
#pragma clang loop unroll(full)
    for (int q = 0; q < 16; ++q)
      d[q] = Accumulate
                 ? Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(d[q], pair(q)))
                 : pair(q);
  } else if constexpr (std::is_same_v<D, float>) {
    // Widening by the pair's factors (1.0 unscaled) splits row pair q into
    // its even lanes, row 2q, and its odd ones, row 2q + 1: one fp32 register
    // each. Two row pairs per iteration, their rows loaded first: on v79 the
    // fastest of the schedules measured, a full unroll and software
    // pipelining included.
#pragma clang loop unroll(disable)
    for (int q = 0; q < 16; q += 2) {
      float *r = dst + 2 * q * ldc;
      HVX_Vector x[4];
      if constexpr (Accumulate) {
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i)
          x[i] = tl_hvx_load<Aligned>(r + i * ldc);
      }
#pragma clang loop unroll(full)
      for (int i = 0; i < 2; ++i) {
        HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(t[q + i], factors(q + i));
        HVX_Vector even = Q6_V_lo_W(w), odd = Q6_V_hi_W(w);
        if constexpr (Accumulate) {
          even = Q6_Vqf32_vadd_Vqf32Vsf(even, x[2 * i]);
          odd = Q6_Vqf32_vadd_Vqf32Vsf(odd, x[2 * i + 1]);
        }
        tl_hvx_store<Aligned>(r + 2 * i * ldc, Q6_Vsf_equals_Vqf32(even));
        tl_hvx_store<Aligned>(r + (2 * i + 1) * ldc, Q6_Vsf_equals_Vqf32(odd));
      }
    }
  } else {
    static_assert(std::is_same_v<D, half>, "fp16 or fp32 matrices");
#pragma clang loop unroll(full)
    for (int q = 0; q < 16; ++q) {
      // Dealing a row pair with itself puts row 2q in both halves of the low
      // register and row 2q + 1 in both halves of the high one.
      HVX_Vector v = pair(q);
      HVX_VectorPair d = Q6_W_vdeal_VVR(v, v, -2);
      half *r0 = dst + 2 * q * ldc;
      tl_crouton_half_row<Accumulate, Aligned>(r0, Q6_V_lo_W(d));
      tl_crouton_half_row<Accumulate, Aligned>(r0 + ldc, Q6_V_hi_W(d));
    }
  }
}

// ---------------------------------------------------------------------------
// HMX weight tiles from 4-bit panels (T.gemm_blockscaled on HMX): 32 rows of
// a 4-bit B [N, K] in the panel layout (tilelang.hexagon.intrinsics.panel)
// become the K / 32 fp16 weight tiles of 32 output columns.
// ---------------------------------------------------------------------------

// The fp16 values of the nibble codes where vlut16 reads them: on v79 entry i
// of table 0 is halfword 2i, so halfwords 2i and 2i + 1 both hold code i.
alignas(128) static const uint16_t tl_crouton_nibble_tables[2][64] = {
    // Unsigned: 0..15.
    {0x0000, 0x0000, 0x3C00, 0x3C00, 0x4000, 0x4000, 0x4200, 0x4200,
     0x4400, 0x4400, 0x4500, 0x4500, 0x4600, 0x4600, 0x4700, 0x4700,
     0x4800, 0x4800, 0x4880, 0x4880, 0x4900, 0x4900, 0x4980, 0x4980,
     0x4A00, 0x4A00, 0x4A80, 0x4A80, 0x4B00, 0x4B00, 0x4B80, 0x4B80},
    // Signed: 0..7, then -8..-1.
    {0x0000, 0x0000, 0x3C00, 0x3C00, 0x4000, 0x4000, 0x4200, 0x4200,
     0x4400, 0x4400, 0x4500, 0x4500, 0x4600, 0x4600, 0x4700, 0x4700,
     0xC800, 0xC800, 0xC700, 0xC700, 0xC600, 0xC600, 0xC500, 0xC500,
     0xC400, 0xC400, 0xC200, 0xC200, 0xC000, 0xC000, 0xBC00, 0xBC00},
};

// Panel registers src[0, R) to the weight-tile registers dst[0, 4R), scaled by
// the row pairs' factors in `scale`. A panel register holds 32 rows (output
// columns) by 8 columns of K, a row's 8 nibbles in one 32-bit lane; its four
// weight-tile registers each hold two columns of K by the 32 rows, interleaved.
// vlut16 looks up the low nibbles of the even bytes into one register and of
// the odd bytes into another, the high nibbles after a shift; vshuffe and
// vshuffo pair the two nibbles of each byte.
template <int R>
TL_DEVICE void tl_crouton_dequant4_block(HVX_Vector *__restrict__ dst,
                                         const HVX_Vector *__restrict__ src,
                                         HVX_Vector table, HVX_Vector scale) {
  HVX_Vector v[R];
  // All loads first, so the stores of one register do not wait behind the
  // loads of the next.
#pragma clang loop unroll(full)
  for (int i = 0; i < R; ++i)
    v[i] = src[i];
#pragma clang loop unroll(full)
  for (int i = 0; i < R; ++i) {
    HVX_VectorPair lo = Q6_Wh_vlut16_VbVhR_nomatch(v[i], table, 0);
    HVX_VectorPair hi =
        Q6_Wh_vlut16_VbVhR_nomatch(Q6_Vuh_vlsr_VuhR(v[i], 4), table, 0);
    const HVX_Vector k[4] = {
        Q6_Vh_vshuffe_VhVh(Q6_V_lo_W(hi), Q6_V_lo_W(lo)), // columns 0, 1
        Q6_Vh_vshuffe_VhVh(Q6_V_hi_W(hi), Q6_V_hi_W(lo)), // 2, 3
        Q6_Vh_vshuffo_VhVh(Q6_V_lo_W(hi), Q6_V_lo_W(lo)), // 4, 5
        Q6_Vh_vshuffo_VhVh(Q6_V_hi_W(hi), Q6_V_hi_W(lo)), // 6, 7
    };
#pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j)
      dst[4 * i + j] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(k[j], scale));
  }
}

// Dequantizes panel registers [v0, v0 + n) of one 32-row panel of a 4-bit B
// into `w`, its weight tiles: register v covers columns [8v, 8v + 8) of K and
// becomes registers 4v..4v + 3 of the tiles, tile kt at w + kt * 1024.
// `sfb` holds the panel's fp16 scales, factor (g, i) for group g of gb
// columns of K and row i at sfb[g * ld_sfb + i]; gb is a multiple of 8. A
// function of its own, as the GEMMs are (TL_GEMM): inlined into a kernel's
// loop it competes for registers and spills its lookup table.
template <bool Signed>
static __attribute__((noinline)) void
tl_crouton_dequant4(half *__restrict__ w, const void *__restrict__ panel,
                    const half *__restrict__ sfb, int ld_sfb, int gb, int v0,
                    int n) {
  const HVX_Vector table =
      *reinterpret_cast<const HVX_Vector *>(tl_crouton_nibble_tables[Signed]);
  const HVX_Vector *src = reinterpret_cast<const HVX_Vector *>(panel);
  HVX_Vector *dst = reinterpret_cast<HVX_Vector *>(w);
  const int per_group = gb >> 3; // panel registers per group of factors
  // One division a call: Hexagon divides in software, and a caller's gb is
  // not always a constant here.
  int g = v0 / per_group;
  for (int v = v0, end = v0 + n; v < end; ++g) {
    const int stop = end < (g + 1) * per_group ? end : (g + 1) * per_group;
    // The group's 32 factors, each in both lanes of its row's pair.
    HVX_Vector s = tl_hvx_load_bytes(sfb + g * ld_sfb, 64);
    s = Q6_V_lo_W(Q6_W_vshuff_VVR(s, s, -2));
    for (; v + 8 <= stop; v += 8)
      tl_crouton_dequant4_block<8>(dst + 4 * v, src + v, table, s);
    for (; v < stop; ++v)
      tl_crouton_dequant4_block<1>(dst + 4 * v, src + v, table, s);
  }
}
