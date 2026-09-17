/*!
 * \file tl/hexagon/op/copy.cc
 * \brief Hexagon implementation for tl.copy lowering.
 *
 * Generic copies reuse LowerNormalCopy.  A full FP16 native HMX Crouton buffer
 * copied to/from a row-major matrix region lowers to one strided pack/unpack
 * helper.  The row-major endpoint may live in DDR or VTCM and may be a matrix
 * slice of a larger tensor.  Common non-transposed layouts use HVX
 * vshuff/vdeal; other layouts retain a scalar correctness fallback.
 */

#include "op/copy.h"
#include "op/utils.h"

#include "backend/common/target_utils.h"

#include <optional>
#include <tvm/tirx/builtin.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace hexagon {

namespace {

enum class HMXLayoutKind : int {
  kActivation = 0,
  kWeight = 1,
  kActivationTransposed = 2,
  kWeightTransposed = 3,
};

PrimExpr CroutonPos(const PrimExpr &row, const PrimExpr &col) {
  return floordiv(floormod(row, 32), 2) * 64 + floormod(col, 32) * 2 +
         floormod(row, 2);
}

PrimExpr FlattenLayoutIndex(const Layout &layout,
                            const Array<PrimExpr> &logical_indices) {
  Array<PrimExpr> physical_indices = layout->Forward(logical_indices);
  Array<PrimExpr> physical_shape = layout->OutputShape();
  ICHECK_EQ(physical_indices.size(), physical_shape.size());
  PrimExpr flat = 0;
  for (size_t i = 0; i < physical_indices.size(); ++i)
    flat = flat * physical_shape[i] + physical_indices[i];
  return flat;
}

std::optional<HMXLayoutKind>
MatchHMXLayout(const Buffer &buffer, const Layout &layout,
               arith::Analyzer *analyzer) {
  if (buffer->shape.size() != 2 || layout->InputDim() != 2)
    return std::nullopt;
  auto input_shape = layout->InputShape();
  if (input_shape.size() != buffer->shape.size())
    return std::nullopt;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (!analyzer->CanProveEqual(input_shape[i], buffer->shape[i]))
      return std::nullopt;
  }

  Var row("_hmx_row", DataType::Int(32));
  Var col("_hmx_col", DataType::Int(32));
  analyzer->Bind(row, Range::FromMinExtent(0, buffer->shape[0]));
  analyzer->Bind(col, Range::FromMinExtent(0, buffer->shape[1]));
  PrimExpr actual = FlattenLayoutIndex(layout, {row, col});
  PrimExpr normal = CroutonPos(row, col);
  PrimExpr transposed = CroutonPos(col, row);
  PrimExpr row_tiles = floordiv(buffer->shape[0], 32);
  PrimExpr col_tiles = floordiv(buffer->shape[1], 32);
  const std::pair<HMXLayoutKind, PrimExpr> candidates[] = {
      {HMXLayoutKind::kActivation,
       (floordiv(row, 32) * col_tiles + floordiv(col, 32)) * 1024 +
           normal},
      {HMXLayoutKind::kWeight,
       (floordiv(col, 32) * row_tiles + floordiv(row, 32)) * 1024 +
           normal},
      {HMXLayoutKind::kActivationTransposed,
       (floordiv(col, 32) * row_tiles + floordiv(row, 32)) * 1024 +
           transposed},
      {HMXLayoutKind::kWeightTransposed,
       (floordiv(row, 32) * col_tiles + floordiv(col, 32)) * 1024 +
           transposed},
  };
  for (const auto &[kind, expected] : candidates) {
    if (analyzer->CanProveEqual(actual, expected))
      return kind;
  }
  return std::nullopt;
}

bool IsFull2DRegion(const Buffer &buffer, const Array<Range> &region,
                    arith::Analyzer *analyzer) {
  if (buffer->shape.size() != 2 || region.size() != 2)
    return false;
  for (size_t i = 0; i < 2; ++i) {
    if (!analyzer->CanProveEqual(region[i]->min, 0) ||
        !analyzer->CanProveEqual(region[i]->extent, buffer->shape[i]))
      return false;
  }
  return true;
}

Array<PrimExpr> GetBufferStrides(const Buffer &buffer) {
  if (!buffer->strides.empty()) {
    ICHECK_EQ(buffer->strides.size(), buffer->shape.size())
        << "Explicit buffer strides must match the buffer rank";
    return buffer->strides;
  }

  Array<PrimExpr> strides;
  strides.resize(buffer->shape.size());
  PrimExpr stride = make_const(buffer->shape.back().dtype(), 1);
  for (int i = static_cast<int>(buffer->shape.size()) - 1; i >= 0; --i) {
    strides.Set(i, stride);
    stride = stride * buffer->shape[i];
  }
  return strides;
}

bool IsMatrixRegion(const Buffer &buffer, const Array<Range> &region,
                    const PrimExpr &rows, const PrimExpr &cols,
                    arith::Analyzer *analyzer) {
  if (buffer->shape.size() < 2 || region.size() != buffer->shape.size())
    return false;
  for (size_t i = 0; i + 2 < region.size(); ++i) {
    if (!analyzer->CanProveEqual(region[i]->extent, 1))
      return false;
  }
  return analyzer->CanProveEqual(region[region.size() - 2]->extent, rows) &&
         analyzer->CanProveEqual(region[region.size() - 1]->extent, cols);
}

PrimExpr MakeAccessPtrAtRegion(const Buffer &buffer, const Array<Range> &region,
                               const Array<PrimExpr> &strides, int rw_mask,
                               const PrimExpr &extent) {
  ICHECK_EQ(region.size(), buffer->shape.size());
  ICHECK_EQ(strides.size(), buffer->shape.size());
  PrimExpr offset = buffer->elem_offset;
  for (size_t i = 0; i < region.size(); ++i)
    offset = offset + region[i]->min * strides[i];

  PrimExpr ptype = tirx::TypeAnnotation(buffer->dtype);
  Array<PrimExpr> args{ptype, buffer->data, offset, extent,
                       IntImm(DataType::Int(32), rw_mask)};
  return Call(DataType::Handle(), builtin::tvm_access_ptr(), args);
}

// Explicit async requests must either become DMA or fail; never silently copy
// synchronously. One call is one FIFO completion unit for T.dma_wait.
Stmt LowerDMACopy(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
  const bool load = IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst);
  const bool store = IsSharedBuffer(op.src) && IsGlobalBuffer(op.dst);
  ICHECK((load || store) && op.src->dtype == op.dst->dtype &&
         op.src->dtype.lanes() == 1 && op.src->dtype.bits() % 8 == 0 &&
         !T.layout_map.count(op.src) && !T.layout_map.count(op.dst))
      << "Hexagon T.dma_copy requires equal scalar byte-addressable dtypes "
         "and row-major global <-> shared buffers (no Crouton layouts)";
  ICHECK(!op.annotations.count("parallel_loop_layout") &&
         !op.annotations.count("coalesced_width"))
      << "Hexagon T.dma_copy does not support SIMT copy hints";

  struct Matrix { PrimExpr rows, cols, stride, span; Array<PrimExpr> strides; };
  auto matrix = [&](const Buffer &buf, const Array<Range> &region) -> Matrix {
    size_t rank = buf->shape.size();
    ICHECK(rank > 0 && region.size() == rank);
    auto strides = GetBufferStrides(buf);
    ICHECK(analyzer->CanProveEqual(strides.back(), 1))
        << "Hexagon T.dma_copy requires unit innermost stride";
    for (size_t i = 0; i < rank; ++i) {
      if (i + 2 < rank)
        ICHECK(analyzer->CanProveEqual(region[i]->extent, 1))
            << "Hexagon T.dma_copy supports only 1D/2D regions";
      ICHECK(analyzer->CanProve(region[i]->min >= 0) &&
             analyzer->CanProve(region[i]->min + region[i]->extent <= buf->shape[i]))
          << "Hexagon T.dma_copy requires provably in-bounds regions";
    }
    PrimExpr cols = region.back()->extent;
    PrimExpr rows = rank == 1 ? PrimExpr(1) : region[rank - 2]->extent;
    PrimExpr stride = rank == 1 ? cols : strides[rank - 2];
    return {rows, cols, stride, (rows - 1) * stride + cols, strides};
  };
  auto src = matrix(op.src, op.src_range);
  auto dst = matrix(op.dst, op.dst_range);
  ICHECK(analyzer->CanProveEqual(src.rows, dst.rows) &&
         analyzer->CanProveEqual(src.cols, dst.cols))
      << "Hexagon T.dma_copy requires matching region shapes";
  auto constant = [&](PrimExpr e) -> int64_t {
    e = analyzer->Simplify(e);
    const auto *v = e.as<IntImmNode>();
    ICHECK(v && v->value > 0)
        << "Hexagon T.dma_copy requires positive static extents and strides";
    return v->value;
  };
  int64_t rows = constant(src.rows), cols = constant(src.cols);
  int64_t ss = constant(src.stride), ds = constant(dst.stride);
  int64_t bytes = op.src->dtype.bytes();
  constexpr int64_t max24 = 0x00ffffff;
  ICHECK(rows <= 65535 && cols <= max24 / bytes && ss >= cols &&
         ds >= cols && ss <= max24 / bytes && ds <= max24 / bytes)
      << "Hexagon T.dma_copy region exceeds DMA descriptor limits";
  PrimExpr sp = MakeAccessPtrAtRegion(op.src, op.src_range, src.strides, 1, src.span);
  PrimExpr dp = MakeAccessPtrAtRegion(op.dst, op.dst_range, dst.strides, 2, dst.span);
  auto u32 = [](int64_t n) { return IntImm(DataType::UInt(32), n); };
  PrimExpr direction = IntImm(DataType::Int(32), load ? 0 : 1);
  if ((rows == 1 || (ss == cols && ds == cols)) && rows * cols * bytes <= max24) {
    return Evaluate(Call(DataType::Int(32), builtin::call_extern(),
        {StringImm("tl_hexagon_dma_async_copy_1d"), dp, sp,
         u32(rows * cols * bytes), direction}));
  }
  return Evaluate(Call(DataType::Int(32), builtin::call_extern(),
      {StringImm("tl_hexagon_dma_async_copy_2d"), dp, sp, u32(ds * bytes),
       u32(ss * bytes), u32(cols * bytes), u32(rows), direction}));
}

std::optional<Stmt> LowerHMXLayoutCopy(const CopyNode &op, const LowerArgs &T,
                                       arith::Analyzer *analyzer) {
  if (op.src->dtype != DataType::Float(16) ||
      op.dst->dtype != DataType::Float(16))
    return std::nullopt;

  std::optional<HMXLayoutKind> src_kind;
  std::optional<HMXLayoutKind> dst_kind;
  if (IsSharedBuffer(op.src) && T.layout_map.count(op.src))
    src_kind = MatchHMXLayout(op.src, T.layout_map[op.src], analyzer);
  if (IsSharedBuffer(op.dst) && T.layout_map.count(op.dst))
    dst_kind = MatchHMXLayout(op.dst, T.layout_map[op.dst], analyzer);

  const bool src_row_major =
      (IsGlobalBuffer(op.src) || IsSharedBuffer(op.src)) &&
      !T.layout_map.count(op.src);
  const bool dst_row_major =
      (IsGlobalBuffer(op.dst) || IsSharedBuffer(op.dst)) &&
      !T.layout_map.count(op.dst);
  const bool pack = dst_kind.has_value() && src_row_major;
  const bool unpack = src_kind.has_value() && dst_row_major;
  if (!pack && !unpack)
    return std::nullopt;

  const Buffer &hmx_buffer = pack ? op.dst : op.src;
  const Array<Range> &hmx_region = pack ? op.dst_range : op.src_range;
  const Buffer &row_buffer = pack ? op.src : op.dst;
  const Array<Range> &row_region = pack ? op.src_range : op.dst_range;
  if (!IsFull2DRegion(hmx_buffer, hmx_region, analyzer))
    return std::nullopt;

  const PrimExpr &rows_expr = hmx_buffer->shape[0];
  const PrimExpr &cols_expr = hmx_buffer->shape[1];
  const auto *rows = rows_expr.as<IntImmNode>();
  const auto *cols = cols_expr.as<IntImmNode>();
  if (rows == nullptr || cols == nullptr || rows->value <= 0 ||
      cols->value <= 0 || rows->value % 32 != 0 || cols->value % 32 != 0 ||
      !IsMatrixRegion(row_buffer, row_region, rows_expr, cols_expr, analyzer))
    return std::nullopt;

  Array<PrimExpr> src_strides = GetBufferStrides(op.src);
  Array<PrimExpr> dst_strides = GetBufferStrides(op.dst);
  Array<PrimExpr> row_strides = pack ? src_strides : dst_strides;
  const size_t row_rank = row_buffer->shape.size();
  PrimExpr matrix_extent = rows_expr * cols_expr;
  PrimExpr src_ptr = MakeAccessPtrAtRegion(op.src, op.src_range, src_strides, 1,
                                          matrix_extent);
  PrimExpr dst_ptr = MakeAccessPtrAtRegion(op.dst, op.dst_range, dst_strides, 2,
                                          matrix_extent);
  int kind = static_cast<int>(pack ? *dst_kind : *src_kind);
  const char *fn = pack ? "tl_hexagon_hmx_pack_crouton"
                        : "tl_hexagon_hmx_unpack_crouton";
  Array<PrimExpr> args = {
      StringImm(fn), dst_ptr, src_ptr, rows_expr, cols_expr,
      row_strides[row_rank - 2], row_strides[row_rank - 1],
      IntImm(DataType::Int(32), kind)};
  return Evaluate(Call(DataType::Int(32), builtin::call_extern(), args));
}

} // namespace

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                               InferLevel level) {
    (void)op;
    (void)T;
    (void)level;
    return {};
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    if (auto async = op.annotations.Get("is_async_copy")) {
      ICHECK(!Downcast<IntImm>(async.value())->value)
          << "Hexagon requires T.dma_copy instead of T.async_copy";
    }
    if (auto async = op.annotations.Get("is_dma_copy")) {
      if (Downcast<IntImm>(async.value())->value)
        return LowerDMACopy(op, T, analyzer);
    }
    if (auto hmx_copy = LowerHMXLayoutCopy(op, T, analyzer))
      return *hmx_copy;
    return LowerNormalCopy(op, T, analyzer);
  }
};

} // namespace hexagon

namespace {

bool MatchHexagonCopyTarget(Target target) { return TargetIsHexagon(target); }

bool RegisterHexagonCopy() {
  RegisterCopyImpl(CopyImpl{
      "hexagon.Copy",
      MatchHexagonCopyTarget,
      100,
      hexagon::Copy::InferLayout,
      hexagon::Copy::Lower,
  });
  return true;
}

const bool hexagon_copy_registered = RegisterHexagonCopy();

} // namespace

} // namespace tl
} // namespace tvm
