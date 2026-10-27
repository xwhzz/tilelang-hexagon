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
#include "utils.h"

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

std::optional<HMXLayoutKind> MatchHMXLayout(const Buffer &buffer,
                                            const Layout &layout,
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
       (floordiv(row, 32) * col_tiles + floordiv(col, 32)) * 1024 + normal},
      {HMXLayoutKind::kWeight,
       (floordiv(col, 32) * row_tiles + floordiv(row, 32)) * 1024 + normal},
      // Prefer the row-tile-major representation when a single tile along an
      // axis makes the two transposed layouts physically equivalent.
      {HMXLayoutKind::kWeightTransposed,
       (floordiv(row, 32) * col_tiles + floordiv(col, 32)) * 1024 + transposed},
      {HMXLayoutKind::kActivationTransposed,
       (floordiv(col, 32) * row_tiles + floordiv(row, 32)) * 1024 + transposed},
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

std::optional<Stmt> LowerHMXLayoutCopy(const CopyNode &op, const LowerArgs &T,
                                       arith::Analyzer *analyzer) {
  if ((op.src->dtype != DataType::Float(16) &&
       op.src->dtype != DataType::Float(32)) ||
      (op.dst->dtype != DataType::Float(16) &&
       op.dst->dtype != DataType::Float(32)))
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
  if (hmx_buffer->dtype != DataType::Float(16))
    return std::nullopt;
  const bool cast = row_buffer->dtype == DataType::Float(32);
  const int kind = static_cast<int>(pack ? *dst_kind : *src_kind);
  if (cast && kind != static_cast<int>(HMXLayoutKind::kActivation))
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
  const char *fn =
      pack ? "tl_hexagon_hmx_pack_crouton" : "tl_hexagon_hmx_unpack_crouton";
  if (cast)
    fn = pack ? "tl_hexagon_hmx_pack_crouton_f32"
              : "tl_hexagon_hmx_unpack_crouton_f32";
  Array<PrimExpr> args = {StringImm(fn),
                          dst_ptr,
                          src_ptr,
                          rows_expr,
                          cols_expr,
                          row_strides[row_rank - 2],
                          row_strides[row_rank - 1],
                          IntImm(DataType::Int(32), kind)};
  if (auto workers = op.annotations.Get("hexagon.copy_workers")) {
    const auto *nw = workers.value().as<IntImmNode>();
    ICHECK(nw && nw->value >= 1 && nw->value <= 6)
        << "hexagon.copy_workers must be an integer in [1, 6]";
    ICHECK(kind == static_cast<int>(HMXLayoutKind::kActivation) ||
           kind == static_cast<int>(HMXLayoutKind::kWeightTransposed))
        << "parallel Crouton copy requires row-tile-major physical storage";
    args.Set(0, StringImm("tl_hexagon_hmx_copy_crouton_parallel"));
    args.push_back(IntImm(DataType::Int(32), (cast ? 2 : 0) + (pack ? 0 : 1)));
    args.push_back(IntImm(DataType::Int(32), nw->value));
  }
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
    if (auto hmx_copy = LowerHMXLayoutCopy(op, T, analyzer))
      return *hmx_copy;
    ICHECK(!op.annotations.count("hexagon.copy_workers"))
        << "hexagon.copy_workers requires a supported full Crouton layout copy";
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
