/*!
 * \file tl/hexagon/op/copy.cc
 * \brief Hexagon implementation for tl.copy lowering.
 *
 * A copy between a Crouton buffer (an HMX operand, CroutonLayout) and a
 * row-major one is one tl_crouton_copy call (crouton.h) per matrix of the
 * region, in the matrix's coordinates: any sub-region, a DDR or VTCM endpoint
 * with any strides, and fp32 <-> fp16 on the way. A region of a stack of
 * matrices loops over the stack's axes. Other copies reuse LowerNormalCopy.
 *
 * A software pipeline's async copy (the injector marks it, and commits and
 * waits for its stage) becomes the descriptors of a User-DMA transfer when
 * DMA can make it and moves it faster than HVX (DMAOutpacesHVX); the
 * pipeline's commit makes a stage's transfers one group. Otherwise it is
 * copied right away, which the pipeline's waits allow.
 */

#include "op/copy.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "utils.h"

#include "backend/common/target_utils.h"
#include "hexagon/layout/crouton.h"
#include "hexagon/op/builtin.h"
#include "hexagon/op/dma.h"

#include <tvm/tirx/builtin.h>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace hexagon {

namespace {

const CroutonLayoutNode *CroutonOf(const Buffer &buffer,
                                   const LayoutMap &layout_map) {
  Optional<Layout> layout = layout_map.Get(buffer);
  return layout ? layout.value().as<CroutonLayoutNode>() : nullptr;
}

std::optional<Stmt> LowerCroutonCopy(const CopyNode &op, const LowerArgs &T,
                                     arith::Analyzer *analyzer) {
  const CroutonLayoutNode *src_crouton = CroutonOf(op.src, T.layout_map);
  const CroutonLayoutNode *dst_crouton = CroutonOf(op.dst, T.layout_map);
  // One Crouton endpoint; Crouton to Crouton copies go element by element.
  if ((src_crouton == nullptr) == (dst_crouton == nullptr))
    return std::nullopt;
  const bool pack = dst_crouton != nullptr;
  const CroutonLayoutNode *crouton = pack ? dst_crouton : src_crouton;
  const Buffer &tiles = pack ? op.dst : op.src;
  const Array<Range> &tile_region = pack ? op.dst_range : op.src_range;
  const Buffer &matrix = pack ? op.src : op.dst;
  const Array<Range> &matrix_region = pack ? op.src_range : op.dst_range;
  if (tiles->dtype != DataType::Float(16) ||
      (matrix->dtype != DataType::Float(16) &&
       matrix->dtype != DataType::Float(32)) ||
      !RowMajor(matrix, T.layout_map))
    return std::nullopt;
  size_t tn = tile_region.size(), mn = matrix_region.size();
  if (tn < 2 || mn < 2)
    return std::nullopt;
  for (int k = 1; k <= 2; ++k) {
    if (!analyzer->CanProveEqual(tile_region[tn - k]->extent,
                                 matrix_region[mn - k]->extent))
      return std::nullopt;
  }
  // The matrices of the region: the axes before the last two, which the two
  // endpoints span alike (a matrix side of rank 2 selects one matrix).
  auto stacked = [](const Array<Range> &region) {
    std::vector<size_t> axes;
    for (size_t i = 0; i + 2 < region.size(); ++i) {
      if (!is_one(region[i]->extent))
        axes.push_back(i);
    }
    return axes;
  };
  std::vector<size_t> tile_axes = stacked(tile_region);
  std::vector<size_t> matrix_axes = stacked(matrix_region);
  if (tile_axes.size() != matrix_axes.size())
    return std::nullopt;
  for (size_t a = 0; a < tile_axes.size(); ++a) {
    if (!analyzer->CanProveEqual(tile_region[tile_axes[a]]->extent,
                                 matrix_region[matrix_axes[a]]->extent))
      return std::nullopt;
  }

  // The matrix's rows and columns in buffer axes: the last two, swapped for
  // a transposed Crouton layout, on both endpoints alike.
  int row_axis = crouton->RowAxis(), col_axis = crouton->ColAxis();
  int last = static_cast<int>(tn) - 1;
  Array<PrimExpr> matrix_strides = GetBufferStrides(matrix);
  PrimExpr rs = matrix_strides[mn - 1 - (last - row_axis)];
  PrimExpr cs = matrix_strides[mn - 1 - (last - col_axis)];

  // One loop per stacked axis; the region's matrix `vars` of the stack.
  std::vector<Var> vars;
  for (size_t a = 0; a < tile_axes.size(); ++a)
    vars.emplace_back("m" + std::to_string(a), DataType::Int(32));
  // A region's matrix at `vars`, from `whole`: the region itself on the last
  // two axes, or, for the Crouton side, the whole matrix from (0, 0).
  auto matrix_at = [&vars](const Array<Range> &region,
                           const std::vector<size_t> &axes, bool whole) {
    Array<Range> one;
    for (size_t i = 0; i < region.size(); ++i) {
      if (i + 2 >= region.size()) {
        one.push_back(
            whole ? Range::FromMinExtent(make_zero(DataType::Int(32)), 1)
                  : region[i]);
        continue;
      }
      auto it = std::find(axes.begin(), axes.end(), i);
      PrimExpr min = region[i]->min;
      if (it != axes.end())
        min = min + vars[it - axes.begin()];
      one.push_back(Range::FromMinExtent(min, 1));
    }
    return one;
  };
  PrimExpr matrix_elems =
      make_const(DataType::Int(32), crouton->Rows() * crouton->Cols());
  PrimExpr tiles_ptr = MakeAccessPtrAtRegion(
      tiles, matrix_at(tile_region, tile_axes, true), GetBufferStrides(tiles),
      pack ? 2 : 1, matrix_elems);
  PrimExpr rows = tile_region[row_axis]->extent;
  PrimExpr cols = tile_region[col_axis]->extent;
  PrimExpr matrix_ptr = MakeAccessPtrAtRegion(
      matrix, matrix_at(matrix_region, matrix_axes, false), matrix_strides,
      pack ? 1 : 2, (rows - 1) * rs + (cols - 1) * cs + 1);
  auto i32 = [](const PrimExpr &e) { return cast(DataType::Int(32), e); };
  Stmt body = Evaluate(
      Call(DataType::Void(), hexagon_crouton_copy(),
           {StringImm(pack ? "true" : "false"),
            StringImm(crouton->column_tiles ? "true" : "false"), tiles_ptr,
            make_const(DataType::Int(32), crouton->RowTiles()),
            make_const(DataType::Int(32), crouton->ColTiles()),
            i32(tile_region[row_axis]->min), i32(tile_region[col_axis]->min),
            i32(rows), i32(cols), matrix_ptr, i32(rs), i32(cs)}));
  for (size_t a = vars.size(); a-- > 0;)
    body = For(vars[a], make_zero(DataType::Int(32)),
               i32(tile_region[tile_axes[a]]->extent), ForKind::kSerial, body);
  return body;
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
    if (auto pipelined =
            op.annotations.Get(attr::kAsyncCopyNoImplicitCommitWait);
        pipelined && !is_zero(Downcast<PrimExpr>(pipelined.value())) &&
        DMAOutpacesHVX(op, analyzer) && DMARejection(op, T, analyzer).empty())
      return LowerDMATransfer(op, T, analyzer, /*commit=*/false);
    if (auto crouton = LowerCroutonCopy(op, T, analyzer))
      return *crouton;
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
