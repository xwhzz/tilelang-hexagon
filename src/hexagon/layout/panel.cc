/*!
 * \file tl/hexagon/layout/panel.cc
 */
#include "hexagon/layout/panel.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

int64_t StaticExtent(const PrimExpr &extent, const char *what) {
  const auto *n = extent.as<IntImmNode>();
  ICHECK(n) << "a panel layout needs a static " << what << ", got " << extent;
  return n->value;
}

// The physical position of every buffer index: [panel, column block, row in
// the panel, column in the lane].
Array<PrimExpr> PanelPosition(const Array<PrimExpr> &shape, int64_t lane) {
  size_t n = shape.size();
  ICHECK_GE(n, 2) << "a panel layout needs at least two dimensions";
  ICHECK(lane == 4 || lane == 8)
      << "a panel lane holds 4 (8-bit) or 8 (4-bit) columns, got " << lane;
  int64_t rows = StaticExtent(shape[n - 2], "row count");
  int64_t cols = StaticExtent(shape[n - 1], "column count");
  ICHECK(rows > 0 && cols > 0 && rows % kPanelRows == 0 && cols % lane == 0)
      << "a panel matrix has rows in multiples of " << kPanelRows
      << " and columns in multiples of " << lane << ", got " << rows << "x"
      << cols;
  PrimExpr batch = make_zero(DataType::Int(32));
  for (size_t i = 0; i + 2 < n; ++i)
    batch = batch * shape[i] + InputPlaceholder(i);
  PrimExpr r = InputPlaceholder(n - 2), c = InputPlaceholder(n - 1);
  PrimExpr p = make_const(DataType::Int(32), kPanelRows);
  PrimExpr l = make_const(DataType::Int(32), lane);
  return {batch * static_cast<int>(rows / kPanelRows) + floordiv(r, p),
          floordiv(c, l), floormod(r, p), floormod(c, l)};
}

} // namespace

PanelLayoutNode::PanelLayoutNode(Array<PrimExpr> shape, int64_t lane)
    : LayoutNode(shape, PanelPosition(shape, lane)), lane(lane) {}

int64_t PanelLayoutNode::Rows() const {
  return InputShape()[InputDim() - 2].as<IntImmNode>()->value;
}

int64_t PanelLayoutNode::Cols() const {
  return InputShape()[InputDim() - 1].as<IntImmNode>()->value;
}

PanelLayout::PanelLayout(Array<PrimExpr> shape, int64_t lane) {
  data_ = make_object<PanelLayoutNode>(shape, lane);
}

void PanelLayoutNode::RegisterReflection() {
  namespace refl = reflection;
  refl::ObjectDef<PanelLayoutNode>().def_ro("lane", &PanelLayoutNode::lane);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  PanelLayoutNode::RegisterReflection();
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.hexagon.PanelLayout",
                        [](Array<PrimExpr> shape, int64_t lane) {
                          return PanelLayout(shape, lane);
                        });
}

} // namespace tl
} // namespace tvm
