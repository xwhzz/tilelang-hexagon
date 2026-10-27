/*!
 * \file tl/hexagon/layout/crouton.cc
 */
#include "hexagon/layout/crouton.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

int64_t StaticExtent(const PrimExpr &extent, const char *what) {
  const auto *n = extent.as<IntImmNode>();
  ICHECK(n) << "a Crouton layout needs a static " << what << ", got " << extent;
  return n->value;
}

// The physical position of every buffer index: [tile, row pair, lane], one
// HVX register per row pair. Leading dimensions pick a matrix, whose tiles
// follow it.
Array<PrimExpr> CroutonPosition(const Array<PrimExpr> &shape, bool column_tiles,
                                bool transposed) {
  size_t n = shape.size();
  ICHECK_GE(n, 2) << "a Crouton layout needs at least two dimensions";
  int row_axis = static_cast<int>(transposed ? n - 1 : n - 2);
  int col_axis = static_cast<int>(transposed ? n - 2 : n - 1);
  int64_t rows = StaticExtent(shape[row_axis], "row count");
  int64_t cols = StaticExtent(shape[col_axis], "column count");
  ICHECK(rows > 0 && cols > 0 && rows % kCroutonTile == 0 &&
         cols % kCroutonTile == 0)
      << "Crouton matrix dimensions must be positive multiples of "
      << kCroutonTile << ", got " << rows << "x" << cols;
  PrimExpr batch = make_zero(DataType::Int(32));
  for (size_t i = 0; i + 2 < n; ++i)
    batch = batch * shape[i] + InputPlaceholder(i);
  PrimExpr r = InputPlaceholder(row_axis), c = InputPlaceholder(col_axis);
  PrimExpr t = make_const(DataType::Int(32), kCroutonTile);
  PrimExpr tile = column_tiles
                      ? floordiv(c, t) * static_cast<int>(rows / kCroutonTile) +
                            floordiv(r, t)
                      : floordiv(r, t) * static_cast<int>(cols / kCroutonTile) +
                            floordiv(c, t);
  int64_t tiles = (rows / kCroutonTile) * (cols / kCroutonTile);
  return {batch * static_cast<int>(tiles) + tile, floordiv(floormod(r, t), 2),
          floormod(c, t) * 2 + floormod(r, 2)};
}

} // namespace

CroutonLayoutNode::CroutonLayoutNode(Array<PrimExpr> shape, bool column_tiles,
                                     bool transposed)
    : LayoutNode(shape, CroutonPosition(shape, column_tiles, transposed)),
      column_tiles(column_tiles), transposed(transposed) {}

int CroutonLayoutNode::RowAxis() const {
  int n = static_cast<int>(InputDim());
  return transposed ? n - 1 : n - 2;
}

int CroutonLayoutNode::ColAxis() const {
  int n = static_cast<int>(InputDim());
  return transposed ? n - 2 : n - 1;
}

int64_t CroutonLayoutNode::Rows() const {
  return InputShape()[RowAxis()].as<IntImmNode>()->value;
}

int64_t CroutonLayoutNode::Cols() const {
  return InputShape()[ColAxis()].as<IntImmNode>()->value;
}

CroutonLayout::CroutonLayout(Array<PrimExpr> shape, bool column_tiles,
                             bool transposed) {
  data_ = make_object<CroutonLayoutNode>(shape, column_tiles, transposed);
}

void CroutonLayoutNode::RegisterReflection() {
  namespace refl = reflection;
  refl::ObjectDef<CroutonLayoutNode>()
      .def_ro("column_tiles", &CroutonLayoutNode::column_tiles)
      .def_ro("transposed", &CroutonLayoutNode::transposed);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  CroutonLayoutNode::RegisterReflection();
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.hexagon.CroutonLayout",
      [](Array<PrimExpr> shape, bool column_tiles, bool transposed) {
        return CroutonLayout(shape, column_tiles, transposed);
      });
}

} // namespace tl
} // namespace tvm
