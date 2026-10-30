/*!
 * \file tl/hexagon/op/dma.cc
 * \brief Explicit Hexagon DMA copy (T.dma_copy).
 *
 * Each T.dma_copy lowers to the descriptors of one asynchronous User-DMA
 * transfer and one commit, so it is one group of the tile program's queue;
 * T.dma_wait(n) waits until at most n groups are pending. An explicit async
 * request either becomes DMA or fails: it never silently falls back to a
 * synchronous copy.
 */

#include "dma.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>

#include <algorithm>
#include <optional>
#include <string>

#include "hexagon/layout/crouton.h"
#include "hexagon/layout/panel.h"
#include "hexagon/op/builtin.h"
#include "hexagon/target_utils.h"
#include "op/utils.h"
#include "support/check.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace hexagon {

namespace {

// A region of a buffer as a matrix of `rows` x `cols` elements whose rows are
// `stride` elements apart: the last dimension, or the last two.
struct Matrix {
  PrimExpr rows, cols, stride;
  Array<PrimExpr> strides;
};

std::optional<Matrix> MatrixOf(const Buffer &buf, const Array<Range> &region,
                               arith::Analyzer *analyzer, std::string *why) {
  size_t rank = buf->shape.size();
  ICHECK(rank > 0 && region.size() == rank);
  Array<PrimExpr> strides = GetBufferStrides(buf);
  if (!analyzer->CanProveEqual(strides.back(), 1)) {
    *why = "Hexagon T.dma_copy requires unit innermost stride";
    return std::nullopt;
  }
  for (size_t i = 0; i < rank; ++i) {
    if (i + 2 < rank && !analyzer->CanProveEqual(region[i]->extent, 1)) {
      *why = "Hexagon T.dma_copy supports only 1D/2D regions";
      return std::nullopt;
    }
    if (!analyzer->CanProve(region[i]->min >= 0) ||
        !analyzer->CanProve(region[i]->min + region[i]->extent <=
                            buf->shape[i])) {
      *why = "Hexagon T.dma_copy requires provably in-bounds regions";
      return std::nullopt;
    }
  }
  PrimExpr cols = region.back()->extent;
  PrimExpr rows = rank == 1 ? PrimExpr(1) : region[rank - 2]->extent;
  PrimExpr stride = rank == 1 ? cols : strides[rank - 2];
  return Matrix{rows, cols, stride, strides};
}

int64_t PositiveConstant(const PrimExpr &e, arith::Analyzer *analyzer) {
  const auto *v = analyzer->Simplify(e).as<IntImmNode>();
  return v && v->value > 0 ? v->value : 0;
}

// Descriptor fields are 24 bits wide, and a 2D descriptor has at most 65535
// rows; larger transfers take several descriptors of one group.
constexpr int64_t kField = 0x00ffffff;
constexpr int64_t kRows = 0xffff;
// 1D chunks stay aligned for every element size.
constexpr int64_t kChunk = int64_t(1) << 23;

} // namespace

const PanelLayoutNode *PanelOf(const Buffer &buffer,
                               const LayoutMap &layout_map) {
  Optional<Layout> layout = layout_map.Get(buffer);
  return layout ? layout.value().as<PanelLayoutNode>() : nullptr;
}

namespace {

// Why the copy between panel buffers `op` cannot move whole panels, or "":
// one matrix on each side, rows on panel boundaries and columns on lane
// boundaries, in bounds.
std::string PanelRejection(const CopyNode &op, const LowerArgs &T,
                           arith::Analyzer *analyzer) {
  const PanelLayoutNode *src = PanelOf(op.src, T.layout_map);
  const PanelLayoutNode *dst = PanelOf(op.dst, T.layout_map);
  if (!src || !dst || src->lane != dst->lane || op.src->dtype != op.dst->dtype)
    return "the copy from " + op.src->name + " to " + op.dst->name +
           " moves a panel layout, so both sides need it, for the same "
           "element width: the other side is packed once, ahead of the kernel";
  const bool load = IsGlobalBuffer(op.src) && IsVTCMBuffer(op.dst);
  const bool store = IsVTCMBuffer(op.src) && IsGlobalBuffer(op.dst);
  if (!(load || store))
    return "panel copies move panels between DDR and VTCM";
  for (const auto &[buffer, region] :
       {std::pair{op.src, op.src_range}, std::pair{op.dst, op.dst_range}}) {
    size_t n = region.size();
    for (size_t i = 0; i < n; ++i) {
      if (i + 2 < n && !analyzer->CanProveEqual(region[i]->extent, 1))
        return "a panel copy moves one matrix at a time";
      if (!analyzer->CanProve(region[i]->min >= 0) ||
          !analyzer->CanProve(region[i]->min + region[i]->extent <=
                              buffer->shape[i]))
        return "a panel copy needs provably in-bounds regions";
    }
    PrimExpr rows = make_const(region[n - 2]->min.dtype(), kPanelRows);
    PrimExpr lane = make_const(region[n - 1]->min.dtype(), src->lane);
    if (!analyzer->CanProveEqual(floormod(region[n - 2]->min, rows), 0) ||
        !analyzer->CanProveEqual(floormod(region[n - 2]->extent, rows), 0) ||
        !analyzer->CanProveEqual(floormod(region[n - 1]->min, lane), 0) ||
        !analyzer->CanProveEqual(floormod(region[n - 1]->extent, lane), 0))
      return "a panel copy covers whole panels: rows on multiples of 32 and "
             "columns on multiples of " +
             std::to_string(src->lane);
  }
  size_t sn = op.src_range.size(), dn = op.dst_range.size();
  if (!analyzer->CanProveEqual(op.src_range[sn - 2]->extent,
                               op.dst_range[dn - 2]->extent) ||
      !analyzer->CanProveEqual(op.src_range[sn - 1]->extent,
                               op.dst_range[dn - 1]->extent) ||
      !op.src_range[sn - 2]->extent.as<IntImmNode>() ||
      !op.src_range[sn - 1]->extent.as<IntImmNode>())
    return "a panel copy needs static, matching region shapes";
  return "";
}

// A column block of a panel, 32 lanes of 32 bits, is one register.
constexpr int64_t kPanelBlockBytes = 128;

// The transfer of `op` between panel buffers: each panel of the region is a
// row of its column blocks, panels a panel's bytes apart on each side.
Stmt LowerPanelTransfer(const CopyNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer, bool commit) {
  const PanelLayoutNode *panel = PanelOf(op.src, T.layout_map);
  const bool load = IsGlobalBuffer(op.src);
  size_t sn = op.src_range.size(), dn = op.dst_range.size();
  int64_t panels =
      op.src_range[sn - 2]->extent.as<IntImmNode>()->value / kPanelRows;
  int64_t row_bytes = op.src_range[sn - 1]->extent.as<IntImmNode>()->value /
                      panel->lane * kPanelBlockBytes;
  auto stride = [&](const Buffer &buffer) {
    return PanelOf(buffer, T.layout_map)->Cols() / panel->lane *
           kPanelBlockBytes;
  };
  int64_t ss = stride(op.src), ds = stride(op.dst);
  ICHECK(row_bytes <= kField && ss <= kField && ds <= kField)
      << "a panel copy's panels must fit the 24-bit DMA descriptor fields";
  int64_t bits = op.src->dtype.bits();
  auto u32 = [](int64_t n) { return IntImm(DataType::UInt(32), n); };
  // A pointer at the region's logical start, `first` panels in: LowerTileOp
  // maps it through the layout to the panel's bytes.
  auto at = [&](const Buffer &buffer, const Array<Range> &region, int rw,
                int64_t first, int64_t count, int64_t panel_stride) {
    Array<Range> start = region;
    size_t r = region.size() - 2;
    start.Set(
        r, Range::FromMinExtent(region[r]->min + IntImm(region[r]->min.dtype(),
                                                        first * kPanelRows),
                                region[r]->extent));
    int64_t span = (count - 1) * panel_stride + row_bytes;
    return MakeAccessPtrAtRegion(buffer, start, GetBufferStrides(buffer), rw,
                                 IntImm(DataType::Int(32), span * 8 / bits));
  };
  PrimExpr direction = IntImm(DataType::Int(32), load ? 0 : 1);
  Array<Stmt> seq;
  // Contiguous panels go in chunks of whole panels.
  bool dense = ss == row_bytes && ds == row_bytes;
  int64_t per = dense ? std::max<int64_t>(1, kChunk / row_bytes) : kRows;
  for (int64_t first = 0; first < panels; first += per) {
    int64_t count = std::min(per, panels - first);
    PrimExpr sp = at(op.src, op.src_range, 1, first, count, ss);
    PrimExpr dp = at(op.dst, op.dst_range, 2, first, count, ds);
    Array<PrimExpr> args =
        dense ? Array<PrimExpr>{dp,
                                sp,
                                u32(count * row_bytes),
                                u32(count * row_bytes),
                                u32(count * row_bytes),
                                u32(1),
                                direction}
              : Array<PrimExpr>{
                    dp,         sp,       u32(ds), u32(ss), u32(row_bytes),
                    u32(count), direction};
    seq.push_back(
        Evaluate(Call(DataType::Int(32), hexagon_dma_submit(), args)));
  }
  if (commit)
    seq.push_back(Evaluate(Call(DataType::Int(32), hexagon_dma_commit(), {})));
  return SeqStmt::Flatten(seq);
}

const CroutonLayoutNode *CroutonOf(const Buffer &buffer,
                                   const LayoutMap &layout_map) {
  Optional<Layout> layout = layout_map.Get(buffer);
  return layout ? layout.value().as<CroutonLayoutNode>() : nullptr;
}

// Why the copy between Crouton buffers `op` cannot move whole tiles, or "": one
// matrix on each side, of the same tile order, its rows and columns on tile
// boundaries, in bounds.
std::string CroutonRejection(const CopyNode &op, const LowerArgs &T,
                             arith::Analyzer *analyzer) {
  const CroutonLayoutNode *src = CroutonOf(op.src, T.layout_map);
  const CroutonLayoutNode *dst = CroutonOf(op.dst, T.layout_map);
  if (!src || !dst || src->column_tiles != dst->column_tiles ||
      src->transposed != dst->transposed || op.src->dtype != op.dst->dtype)
    return "the copy from " + op.src->name + " to " + op.dst->name +
           " moves Crouton tiles, so both sides need the same Crouton layout "
           "(tile order and transpose) and dtype";
  const bool load = IsGlobalBuffer(op.src) && IsVTCMBuffer(op.dst);
  const bool store = IsVTCMBuffer(op.src) && IsGlobalBuffer(op.dst);
  if (!(load || store))
    return "Crouton copies move tiles between DDR and VTCM";
  for (const auto &[buffer, region] :
       {std::pair{op.src, op.src_range}, std::pair{op.dst, op.dst_range}}) {
    size_t n = region.size();
    for (size_t i = 0; i < n; ++i) {
      if (i + 2 < n && !analyzer->CanProveEqual(region[i]->extent, 1))
        return "a Crouton copy moves one matrix at a time";
      if (!analyzer->CanProve(region[i]->min >= 0) ||
          !analyzer->CanProve(region[i]->min + region[i]->extent <=
                              buffer->shape[i]))
        return "a Crouton copy needs provably in-bounds regions";
      if (i + 2 >= n) {
        PrimExpr tile = make_const(region[i]->min.dtype(), kCroutonTile);
        if (!analyzer->CanProveEqual(floormod(region[i]->min, tile), 0) ||
            !analyzer->CanProveEqual(floormod(region[i]->extent, tile), 0))
          return "a Crouton copy covers whole tiles: rows and columns on "
                 "multiples of 32";
      }
    }
  }
  size_t sn = op.src_range.size(), dn = op.dst_range.size();
  for (size_t i = 1; i <= 2; ++i) {
    if (!op.src_range[sn - i]->extent.as<IntImmNode>() ||
        !analyzer->CanProveEqual(op.src_range[sn - i]->extent,
                                 op.dst_range[dn - i]->extent))
      return "a Crouton copy needs static, matching region shapes";
  }
  return "";
}

// The transfer of `op` between Crouton buffers. Tiles follow each other along
// the matrix's rows (column tiles: an HMX weight's K) or its columns (row
// tiles), so the region's tiles along that axis are one contiguous run, and
// its runs along the other axis are the matrix's runs apart on each side.
Stmt LowerCroutonTransfer(const CopyNode &op, const LowerArgs &T,
                          arith::Analyzer *analyzer, bool commit) {
  const CroutonLayoutNode *src = CroutonOf(op.src, T.layout_map);
  const CroutonLayoutNode *dst = CroutonOf(op.dst, T.layout_map);
  const bool load = IsGlobalBuffer(op.src);
  // A buffer's axis along which its tiles run, and the one across its runs.
  auto run_axis = [](const CroutonLayoutNode *layout) {
    return layout->column_tiles ? layout->RowAxis() : layout->ColAxis();
  };
  auto across_axis = [](const CroutonLayoutNode *layout) {
    return layout->column_tiles ? layout->ColAxis() : layout->RowAxis();
  };
  auto extent = [](const Array<Range> &region, int axis) {
    return region[axis]->extent.as<IntImmNode>()->value;
  };
  int64_t tile_bytes = kCroutonTileElems * op.src->dtype.bytes();
  int64_t runs = extent(op.src_range, across_axis(src)) / kCroutonTile;
  int64_t run_bytes =
      extent(op.src_range, run_axis(src)) / kCroutonTile * tile_bytes;
  auto stride = [&](const CroutonLayoutNode *layout) {
    return (layout->column_tiles ? layout->RowTiles() : layout->ColTiles()) *
           tile_bytes;
  };
  int64_t ss = stride(src), ds = stride(dst);
  ICHECK(run_bytes <= kField && ss <= kField && ds <= kField)
      << "a Crouton copy's runs of tiles must fit the 24-bit DMA descriptor "
         "fields";
  int64_t bytes = op.src->dtype.bytes();
  auto u32 = [](int64_t n) { return IntImm(DataType::UInt(32), n); };
  // A pointer at the region's logical start, `first` runs in: LowerTileOp
  // maps it through the layout to the tile's bytes.
  auto at = [&](const Buffer &buffer, const CroutonLayoutNode *layout,
                const Array<Range> &region, int rw, int64_t first,
                int64_t count, int64_t run_stride) {
    Array<Range> start = region;
    int axis = across_axis(layout);
    start.Set(axis, Range::FromMinExtent(region[axis]->min +
                                             IntImm(region[axis]->min.dtype(),
                                                    first * kCroutonTile),
                                         region[axis]->extent));
    int64_t span = (count - 1) * run_stride + run_bytes;
    return MakeAccessPtrAtRegion(buffer, start, GetBufferStrides(buffer), rw,
                                 IntImm(DataType::Int(32), span / bytes));
  };
  PrimExpr direction = IntImm(DataType::Int(32), load ? 0 : 1);
  Array<Stmt> seq;
  // One run, or runs that follow each other on both sides, go in chunks of
  // whole runs.
  bool dense = runs == 1 || (ss == run_bytes && ds == run_bytes);
  int64_t per = dense ? std::max<int64_t>(1, kChunk / run_bytes) : kRows;
  for (int64_t first = 0; first < runs; first += per) {
    int64_t count = std::min(per, runs - first);
    PrimExpr sp = at(op.src, src, op.src_range, 1, first, count, ss);
    PrimExpr dp = at(op.dst, dst, op.dst_range, 2, first, count, ds);
    Array<PrimExpr> args =
        dense ? Array<PrimExpr>{dp,
                                sp,
                                u32(count * run_bytes),
                                u32(count * run_bytes),
                                u32(count * run_bytes),
                                u32(1),
                                direction}
              : Array<PrimExpr>{
                    dp,         sp,       u32(ds), u32(ss), u32(run_bytes),
                    u32(count), direction};
    seq.push_back(
        Evaluate(Call(DataType::Int(32), hexagon_dma_submit(), args)));
  }
  if (commit)
    seq.push_back(Evaluate(Call(DataType::Int(32), hexagon_dma_commit(), {})));
  return SeqStmt::Flatten(seq);
}

} // namespace

std::string DMARejection(const CopyNode &op, const LowerArgs &T,
                         arith::Analyzer *analyzer) {
  if (PanelOf(op.src, T.layout_map) || PanelOf(op.dst, T.layout_map))
    return PanelRejection(op, T, analyzer);
  if (CroutonOf(op.src, T.layout_map) || CroutonOf(op.dst, T.layout_map))
    return CroutonRejection(op, T, analyzer);
  const bool load = IsGlobalBuffer(op.src) && IsVTCMBuffer(op.dst);
  const bool store = IsVTCMBuffer(op.src) && IsGlobalBuffer(op.dst);
  if (!(load || store) || op.src->dtype != op.dst->dtype ||
      op.src->dtype.lanes() != 1 || op.src->dtype.bits() % 8 != 0 ||
      T.layout_map.count(op.src) || T.layout_map.count(op.dst))
    return "Hexagon T.dma_copy requires equal scalar byte-addressable dtypes "
           "and row-major global <-> shared buffers, or panel or Crouton "
           "layouts on both sides";
  if (op.annotations.count("parallel_loop_layout") ||
      op.annotations.count("coalesced_width"))
    return "Hexagon T.dma_copy does not support SIMT copy hints";
  std::string why;
  auto src = MatrixOf(op.src, op.src_range, analyzer, &why);
  auto dst = src ? MatrixOf(op.dst, op.dst_range, analyzer, &why) : src;
  if (!src || !dst)
    return why;
  if (!analyzer->CanProveEqual(src->rows, dst->rows) ||
      !analyzer->CanProveEqual(src->cols, dst->cols))
    return "Hexagon T.dma_copy requires matching region shapes";
  int64_t rows = PositiveConstant(src->rows, analyzer);
  int64_t cols = PositiveConstant(src->cols, analyzer);
  int64_t ss = PositiveConstant(src->stride, analyzer);
  int64_t ds = PositiveConstant(dst->stride, analyzer);
  if (!rows || !cols || !ss || !ds)
    return "Hexagon T.dma_copy requires positive static extents and strides";
  if (ss < cols || ds < cols)
    return "Hexagon T.dma_copy requires rows that do not overlap";
  int64_t bytes = op.src->dtype.bytes();
  if (rows == 1 || (ss == cols && ds == cols))
    return "";
  if (cols * bytes > kField || ss * bytes > kField || ds * bytes > kField)
    return "Hexagon T.dma_copy rows and strides must fit the 24-bit DMA "
           "descriptor fields (16 MiB)";
  // The runtime builds the v75+ type-9 2D descriptor; older architectures
  // would need another descriptor format.
  const int arch = HexagonArchVersion(T.target);
  if (arch != 0 && arch < 75)
    return "2D T.dma_copy needs Hexagon v75 or newer; the target is hexagonv" +
           std::to_string(arch) +
           ". Copy contiguous rows with 1D transfers instead.";
  return "";
}

bool DMAOutpacesHVX(const CopyNode &op, arith::Analyzer *analyzer) {
  constexpr int64_t kMinRowBytes = 512;
  // The row length, and whether the rows of a buffer's region follow each
  // other in memory.
  const Array<Range> &region = op.src_range;
  int64_t cols = PositiveConstant(region.back()->extent, analyzer);
  int64_t rows =
      region.size() == 1
          ? 1
          : PositiveConstant(region[region.size() - 2]->extent, analyzer);
  if (!rows || !cols)
    return false;
  auto dense = [&](const Buffer &buffer) {
    Array<PrimExpr> strides = GetBufferStrides(buffer);
    return strides.size() < 2 ||
           analyzer->CanProveEqual(strides[strides.size() - 2],
                                   IntImm(DataType::Int(64), cols));
  };
  bool contiguous = rows == 1 || (dense(op.src) && dense(op.dst));
  return contiguous || cols * op.src->dtype.bytes() >= kMinRowBytes;
}

Stmt LowerDMATransfer(const CopyNode &op, const LowerArgs &T,
                      arith::Analyzer *analyzer, bool commit) {
  std::string why = DMARejection(op, T, analyzer);
  ICHECK(why.empty()) << why;
  if (PanelOf(op.src, T.layout_map))
    return LowerPanelTransfer(op, T, analyzer, commit);
  if (CroutonOf(op.src, T.layout_map))
    return LowerCroutonTransfer(op, T, analyzer, commit);
  const bool load = IsGlobalBuffer(op.src);
  std::string unused;
  Matrix src = *MatrixOf(op.src, op.src_range, analyzer, &unused);
  Matrix dst = *MatrixOf(op.dst, op.dst_range, analyzer, &unused);
  int64_t rows = PositiveConstant(src.rows, analyzer);
  int64_t cols = PositiveConstant(src.cols, analyzer);
  int64_t ss = PositiveConstant(src.stride, analyzer);
  int64_t ds = PositiveConstant(dst.stride, analyzer);
  int64_t bytes = op.src->dtype.bytes();
  auto u32 = [](int64_t n) { return IntImm(DataType::UInt(32), n); };
  PrimExpr direction = IntImm(DataType::Int(32), load ? 0 : 1);
  // One descriptor of `count` rows starting `src_offset`/`dst_offset` elements
  // into the regions. Strides and row sizes are in bytes.
  auto submit = [&](int64_t src_offset, int64_t dst_offset, int64_t count,
                    int64_t row_bytes, int64_t src_stride, int64_t dst_stride) {
    int64_t src_span = (count - 1) * src_stride + row_bytes;
    int64_t dst_span = (count - 1) * dst_stride + row_bytes;
    PrimExpr sp = MakeAccessPtrAtRegion(
        op.src, op.src_range, src.strides, 1,
        IntImm(DataType::Int(32), src_span / bytes), src_offset);
    PrimExpr dp = MakeAccessPtrAtRegion(
        op.dst, op.dst_range, dst.strides, 2,
        IntImm(DataType::Int(32), dst_span / bytes), dst_offset);
    return Evaluate(Call(DataType::Int(32), hexagon_dma_submit(),
                         {dp, sp, u32(dst_stride), u32(src_stride),
                          u32(row_bytes), u32(count), direction}));
  };
  Array<Stmt> seq;
  if (rows == 1 || (ss == cols && ds == cols)) {
    int64_t total = rows * cols * bytes;
    for (int64_t done = 0; done < total; done += kChunk) {
      int64_t n = std::min(kChunk, total - done);
      seq.push_back(submit(done / bytes, done / bytes, 1, n, n, n));
    }
  } else {
    for (int64_t row = 0; row < rows; row += kRows) {
      seq.push_back(submit(row * ss, row * ds, std::min(kRows, rows - row),
                           cols * bytes, ss * bytes, ds * bytes));
    }
  }
  if (commit)
    seq.push_back(Evaluate(Call(DataType::Int(32), hexagon_dma_commit(), {})));
  return SeqStmt::Flatten(seq);
}

Stmt LowerCacheWriteback(const Buffer &buffer, const Array<Range> &region,
                         const LowerArgs &T, arith::Analyzer *analyzer) {
  ICHECK(IsGlobalBuffer(buffer))
      << "a cache writeback is of DDR, not of " << buffer->name;
  size_t n = region.size();
  for (size_t i = 0; i + 2 < n; ++i)
    ICHECK(analyzer->CanProveEqual(region[i]->extent, 1))
        << "a cache writeback covers one matrix of " << buffer->name
        << " at a time";
  int64_t bytes = buffer->dtype.bytes();
  Array<PrimExpr> strides = GetBufferStrides(buffer);
  auto u32 = [&](const PrimExpr &e) {
    return cast(DataType::UInt(32), analyzer->Simplify(e));
  };
  // `value` in the dtype of `like`.
  auto as = [](const PrimExpr &like, int64_t value) {
    return make_const(like.dtype(), value);
  };
  auto writeback = [&](const Array<Range> &start, const PrimExpr &run_bytes,
                       const PrimExpr &runs, const PrimExpr &stride) {
    PrimExpr span = (runs - 1) * stride + run_bytes;
    PrimExpr ptr = MakeAccessPtrAtRegion(
        buffer, start, strides, kAccessRead,
        analyzer->Simplify(floordiv(span, as(span, bytes))));
    return Evaluate(Call(DataType::Int(32), hexagon_cache_writeback(),
                         {ptr, u32(run_bytes), u32(runs), u32(stride)}));
  };
  if (const CroutonLayoutNode *layout = CroutonOf(buffer, T.layout_map)) {
    int run_axis = layout->column_tiles ? layout->RowAxis() : layout->ColAxis();
    int across_axis =
        layout->column_tiles ? layout->ColAxis() : layout->RowAxis();
    // The tiles the region touches along an axis, [first, end).
    auto tile = [&](int axis) {
      return make_const(region[axis]->min.dtype(), kCroutonTile);
    };
    auto first = [&](int axis) {
      return floordiv(region[axis]->min, tile(axis));
    };
    auto end = [&](int axis) {
      return floordiv(region[axis]->min + region[axis]->extent + tile(axis) - 1,
                      tile(axis));
    };
    int64_t tile_bytes = kCroutonTileElems * bytes;
    int64_t stride =
        (layout->column_tiles ? layout->RowTiles() : layout->ColTiles()) *
        tile_bytes;
    // The first tile's first element: LowerTileOp maps it through the layout.
    Array<Range> start = region;
    for (int axis : {run_axis, across_axis})
      start.Set(axis, Range::FromMinExtent(first(axis) * tile(axis),
                                           region[axis]->extent));
    PrimExpr run_tiles = end(run_axis) - first(run_axis);
    PrimExpr run_bytes = run_tiles * as(run_tiles, tile_bytes);
    return writeback(start, run_bytes, end(across_axis) - first(across_axis),
                     make_const(run_bytes.dtype(), stride));
  }
  ICHECK(RowMajor(buffer, T.layout_map))
      << "a cache writeback of " << buffer->name
      << " needs a row-major or Crouton layout";
  // The region's rows of its last axis, at the buffer's row stride.
  PrimExpr row_bytes = region[n - 1]->extent * as(region[n - 1]->extent, bytes);
  PrimExpr rows = n >= 2 ? region[n - 2]->extent : as(row_bytes, 1);
  PrimExpr stride =
      n >= 2 ? strides[n - 2] * as(strides[n - 2], bytes) : row_bytes;
  return writeback(region, row_bytes, rows, stride);
}

const VarNode *DMAPointedVar(const PrimExpr &ptr) {
  const auto *call = ptr.as<CallNode>();
  if (call == nullptr)
    return ptr.as<VarNode>();
  if (call->op.same_as(builtin::tvm_access_ptr()))
    return call->args[1].as<VarNode>();
  if (call->op.same_as(builtin::address_of()))
    if (const auto *load = call->args[0].as<BufferLoadNode>())
      return load->buffer->data.get();
  return nullptr;
}

namespace {

void CheckHexagonTarget(const Target &target) {
  ICHECK(TargetIsHexagon(target))
      << "T.dma_copy is supported only on Hexagon, got target "
      << target->str();
}

} // namespace

} // namespace hexagon

HexagonDmaCopy::HexagonDmaCopy(Array<PrimExpr> args,
                               Map<String, ObjectRef> annotations) {
  ObjectPtr<HexagonDmaCopyNode> node = make_object<HexagonDmaCopyNode>();
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->src = src_access.region->buffer;
  node->dst = dst_access.region->buffer;
  node->src_range = src_access.region->region;
  node->dst_range = dst_access.region->region;
  node->SetAccessRegions({src_access, dst_access});
  node->annotations = annotations;
  data_ = std::move(node);
}

TileOperator HexagonDmaCopyNode::Clone() const {
  auto op = make_object<HexagonDmaCopyNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = Downcast<ParallelOp>(par_op_->Clone());
  }
  return HexagonDmaCopy(op);
}

LayoutMap HexagonDmaCopyNode::InferLayout(const LayoutInferArgs &layout_args,
                                          InferLevel level) const {
  (void)level;
  hexagon::CheckHexagonTarget(layout_args.target);
  // DMA moves row-major bytes; it neither requires nor proposes a layout.
  return {};
}

Stmt HexagonDmaCopyNode::Lower(const LowerArgs &lower_args,
                               arith::Analyzer *analyzer) const {
  hexagon::CheckHexagonTarget(lower_args.target);
  return hexagon::LowerDMATransfer(*this, lower_args, analyzer,
                                   /*commit=*/true);
}

TIR_REGISTER_TL_TILE_OP(HexagonDmaCopy, dma_copy)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

HexagonCacheWriteback::HexagonCacheWriteback(
    Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  (void)annotations;
  ObjectPtr<HexagonCacheWritebackNode> node =
      make_object<HexagonCacheWritebackNode>();
  // It reads the lines and makes DDR hold what they hold: ordered with the
  // writes before it and the DMA loads after it.
  auto access = NormalizeToAccessRegion(args[0], kAccessReadWrite);
  node->buffer = access.region->buffer;
  node->region = access.region->region;
  node->SetAccessRegions({access});
  data_ = std::move(node);
}

TileOperator HexagonCacheWritebackNode::Clone() const {
  return HexagonCacheWriteback(make_object<HexagonCacheWritebackNode>(*this));
}

LayoutMap
HexagonCacheWritebackNode::InferLayout(const LayoutInferArgs &layout_args,
                                       InferLevel level) const {
  (void)level;
  hexagon::CheckHexagonTarget(layout_args.target);
  return {};
}

Stmt HexagonCacheWritebackNode::Lower(const LowerArgs &lower_args,
                                      arith::Analyzer *analyzer) const {
  hexagon::CheckHexagonTarget(lower_args.target);
  return hexagon::LowerCacheWriteback(buffer, region, lower_args, analyzer);
}

TIR_REGISTER_TL_TILE_OP(HexagonCacheWriteback, cache_writeback)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  HexagonDmaCopyNode::RegisterReflection();
  HexagonCacheWritebackNode::RegisterReflection();
  namespace refl = tvm::ffi::reflection;
  // The commit and wait of a software pipeline's async copies (the target's
  // "async_copy_commit" and "async_copy_wait"): DMA groups of the queue.
  refl::GlobalDef()
      .def("tl.hexagon.AsyncCopyCommit",
           []() -> Stmt {
             return Evaluate(Call(DataType::Int(32), hexagon_dma_commit(), {}));
           })
      .def("tl.hexagon.AsyncCopyWait", [](PrimExpr pending) -> Stmt {
        return Evaluate(Call(DataType::Int(32), hexagon_dma_wait(), {pending}));
      });
}

} // namespace tl
} // namespace tvm
